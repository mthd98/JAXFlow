# jaxflow/optimizers/base_optimizer.py
# --------------------------------------------------
import jax
import jax.numpy as jnp
import optax
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
from jaxflow.core.auto_name import AutoNameMixin

class BaseOptimizer(AutoNameMixin, ABC):
    """
    Base optimizer for JAXFlow.

    A flexible, extensible Optax-based optimizer wrapper providing advanced features
    such as gradient clipping, decoupled weight decay, gradient accumulation,
    mixed-precision loss scaling, and Exponential Moving Average (EMA) tracking.

    Subclass to implement specific optimizers (e.g., Adam, SGD) by defining
    `_create_optax_transform()`.

    Args:
        learning_rate (float): Base learning rate.
        weight_decay (float, optional): Decoupled weight decay rate (AdamW style). Default: 0.0.
        clipnorm (float, optional): Per-gradient norm clip threshold. Default: None.
        global_clipnorm (float, optional): Global gradient norm clip threshold. Default: None.
        loss_scale (float, optional): Loss scale for mixed-precision training. Default: None.
        accumulate_steps (int, optional): Number of steps to accumulate gradients before update. Default: None.
        use_ema (bool, optional): Whether to track an Exponential Moving Average of parameters. Default: False.
        ema_decay (float, optional): Decay rate for EMA. Default: 0.999.
        ema_every (int, optional): Swap to EMA params every N steps (if set). Default: None.
        **kwargs: Additional keyword arguments forwarded to optimizer-specific configuration.

    Attributes:
        learning_rate (float): The base learning rate.
        weight_decay (float): Weight decay hyperparameter.
        clipnorm (float | None): Per-gradient clipping threshold.
        global_clipnorm (float | None): Global gradient norm threshold.
        loss_scale (float | None): Loss scaling factor for mixed-precision.
        accumulate_steps (int | None): Number of steps to accumulate gradients.
        use_ema (bool): Whether EMA is used.
        ema_decay (float): EMA decay rate.
        ema_every (int | None): Swap interval for EMA params.
        config (dict): Additional config passed to the subclass optimizer.
        opt_transform (optax.GradientTransformation): Composed optax transformation pipeline.

    Methods:
        init(params) -> dict:
            Initialize optimizer state for a set of model parameters.
        update(grads, state, params) -> (new_params, new_state):
            Apply an update step given gradients, previous state, and params.
        get_ema_params(state) -> Any | None:
            Retrieve EMA parameters from state, if EMA is enabled.
        get_step(state) -> int:
            Return the step counter from the state.
        reset_accumulation(state, params) -> dict:
            Reset accumulated gradients in the state (for gradient accumulation).
        get_config() -> dict:
            Return the optimizer configuration as a serializable dictionary.

    Example:
        ```python
        class Adam(BaseOptimizer):
            def _create_optax_transform(self, learning_rate, **kwargs):
                return optax.adam(learning_rate)

        optimizer = Adam(learning_rate=1e-3)
        state = optimizer.init(params)
        for batch in dataset:
            grads = compute_grads(params, batch)
            params, state = optimizer.update(grads, state, params)
        ```
    Notes:
        - By default, update returns `(new_params, new_state)`; you never call `optax.apply_updates` directly.
        - Supports gradient accumulation, loss scaling, and parameter averaging for robust, scalable training.
        - All transformations are composed and applied in sequence:
          gradient clipping → weight decay → optimizer.

    Raises:
        NotImplementedError: If subclass does not implement `_create_optax_transform`.
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
