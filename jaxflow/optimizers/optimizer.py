# jaxflow/optimizers/base_optimizer.py
# --------------------------------------------------
import jax
import jax.numpy as jnp
import optax
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
from jaxflow.core.auto_name import AutoNameMixin

class BaseOptimizer(AutoNameMixin,ABC):
    """
    Framework-level wrapper around Optax that adds clipping, weight-decay,
    grad-accumulation, loss-scaling, and EMA.

    NEW (2025-05-23)
    ----------------
    • update() now returns **(new_params, new_state)** rather than (updates, …).
      That means caller code never has to call `optax.apply_updates` itself.

    Public API
    ----------
    init(params)                 -> state
    update(grads, state, params) -> (new_params, new_state)
    get_ema_params(state)        -> params | None
    get_step(state)              -> int
    reset_accumulation(state,
                        params)  -> state
    """

    # -------------------------------------------------- #
    # Construction
    # -------------------------------------------------- #
    def __init__(
        self,
        learning_rate: float,
        *,
        weight_decay: float = 0.0,
        clipnorm: float | None = None,
        global_clipnorm: float | None = None,
        loss_scale: float | None = None,
        accumulate_steps: int | None = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_every: int | None = None,
        
        **kwargs,
    ):
        # Hyper-parameters
        self.learning_rate   = learning_rate
        self.weight_decay    = weight_decay
        self.clipnorm        = clipnorm
        self.global_clipnorm = global_clipnorm
        self.loss_scale      = loss_scale
        self.accumulate_steps = accumulate_steps
        self.use_ema         = use_ema
        self.ema_decay       = ema_decay
        self.ema_every       = ema_every
        self.config          = kwargs

        # Build the core Optax pipeline once
        self._build_optax()

    # -------------------------------------------------- #
    # Optax transform builder
    # -------------------------------------------------- #
    def _build_optax(self):
        txs: list[optax.GradientTransformation] = []

        # 1) Gradient clipping
        if self.global_clipnorm is not None:
            txs.append(optax.clip_by_global_norm(self.global_clipnorm))
        elif self.clipnorm is not None:
            txs.append(optax.clip(self.clipnorm))

        # 2) Decoupled weight decay
        if self.weight_decay and self.weight_decay > 0.0:
            txs.append(optax.add_decayed_weights(self.weight_decay))

        # 3) Base optimizer (provided by subclass)
        base_tx = self._create_optax_transform(self.learning_rate, **self.config)
        txs.append(base_tx)

        # Final chain
        self.opt_transform: optax.GradientTransformation = optax.chain(*txs)

    # -------------------------------------------------- #
    # Must-override: create the base transform (Adam, SGD…)
    # -------------------------------------------------- #
    @abstractmethod
    def _create_optax_transform(
        self, learning_rate: float, **config
    ) -> optax.GradientTransformation:
        ...

    # -------------------------------------------------- #
    # Interface used by Model / training loop
    # -------------------------------------------------- #
    def init(self, params) -> Dict[str, Any]:
        """Return a state-dict compatible with `update`."""
        state = {
            "optax_state": self.opt_transform.init(params),
            "step": 0,
        }

        if self.accumulate_steps and self.accumulate_steps > 1:
            state["accum_grads"] = jax.tree_map(jnp.zeros_like, params)

        if self.use_ema:
            state["ema_params"] = params

        return state

    # -------------------------------------------------- #
    # Core update step  (returns **new_params**!)
    # -------------------------------------------------- #
    def update(
        self,
        grads,
        opt_state: Dict[str, Any],
        params,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Args
        ----
        grads      : PyTree of gradients
        opt_state  : state‐dict previously returned by `init` / `update`
        params     : current model parameters

        Returns
        -------
        (new_params, new_state)
        """
        step = opt_state["step"] + 1
        optax_state = opt_state["optax_state"]

        # ---- loss scaling (mixed precision) ---------------------------
        if self.loss_scale is not None:
            grads = jax.tree_map(lambda g: g / self.loss_scale, grads)

        # ---- gradient accumulation ------------------------------------
        if self.accumulate_steps and self.accumulate_steps > 1:
            acc = opt_state["accum_grads"]
            acc = jax.tree_map(lambda a, g: a + g, acc, grads)

            # Not time to apply yet → just store and exit
            if step % self.accumulate_steps != 0:
                new_state = {
                    **opt_state,
                    "step": step,
                    "accum_grads": acc,
                }
                return params, new_state  # params unchanged

            # Time to update → average & reset accumulator
            grads = jax.tree_map(lambda a: a / self.accumulate_steps, acc)
            reset_acc = jax.tree_map(jnp.zeros_like, acc)
        # ----------------------------------------------------------------

        # ---- optax update ---------------------------------------------
        updates, new_optax_state = self.opt_transform.update(
            grads, optax_state, params
        )
        new_params = optax.apply_updates(params, updates)
        # ----------------------------------------------------------------

        # ---- build new state dict -------------------------------------
        new_state = {
            "optax_state": new_optax_state,
            "step": step,
        }

        if self.accumulate_steps and self.accumulate_steps > 1:
            new_state["accum_grads"] = reset_acc

        # ---- EMA maintenance ------------------------------------------
        if self.use_ema:
            ema_params = opt_state["ema_params"]
            new_ema = jax.tree_map(
                lambda e, p: self.ema_decay * e + (1.0 - self.ema_decay) * p,
                ema_params,
                new_params,
            )
            new_state["ema_params"] = new_ema

            # Optional parameter swap every `ema_every` steps
            if self.ema_every and step % self.ema_every == 0:
                new_params = new_ema  # do the swap

        return new_params, new_state

    # -------------------------------------------------- #
    # Convenience helpers
    # -------------------------------------------------- #
    def get_ema_params(self, state: Dict[str, Any]) -> Optional[Any]:
        return state.get("ema_params") if self.use_ema else None

    def get_step(self, state: Dict[str, Any]) -> int:
        return state["step"]

    def reset_accumulation(self, state: Dict[str, Any], params) -> Dict[str, Any]:
        if not (self.accumulate_steps and self.accumulate_steps > 1):
            return state
        new_state = state.copy()
        new_state["accum_grads"] = jax.tree_map(jnp.zeros_like, params)
        return new_state

    # -------------------------------------------------- #
    # (De)serialisation
    # -------------------------------------------------- #
    def get_config(self) -> Dict[str, Any]:
        cfg = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "clipnorm": self.clipnorm,
            "global_clipnorm": self.global_clipnorm,
            "loss_scale": self.loss_scale,
            "accumulate_steps": self.accumulate_steps,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "ema_every": self.ema_every,
        }
        cfg.update(self.config)
        return cfg
