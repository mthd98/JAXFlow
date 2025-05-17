import jax
import jax.numpy as jnp
import optax
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Pure-functional BaseOptimizer wrapping Optax, with support for:
      - Gradient clipping (per-variable or global norm)
      - Decoupled weight decay
      - Loss scaling for mixed precision
      - Gradient accumulation
      - Exponential Moving Average (EMA) of parameters

    train_step and eval_step functions can be jitted without side-effects.

    The API:
      opt_state, accum_grads, ema_params, step = optimizer.init(params)
      new_params, opt_state, accum_grads, ema_params, step, loss = \
          optimizer.update(params, grads, opt_state, accum_grads, ema_params, step)
    """
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
        clipnorm: float = None,
        global_clipnorm: float = None,
        loss_scale: float = None,
        accumulate_steps: int = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_every: int = None,
        **kwargs,
    ):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.global_clipnorm = global_clipnorm
        self.loss_scale = loss_scale
        self.accumulate_steps = accumulate_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_every = ema_every
        self.config = kwargs

        # Build core optax transform
        self._build_optax()

    def _build_optax(self):
        transforms = []
        # Clipping
        if self.global_clipnorm is not None:
            transforms.append(optax.clip_by_global_norm(self.global_clipnorm))
        elif self.clipnorm is not None:
            transforms.append(optax.clip(self.clipnorm))
        # Decoupled weight decay
        if self.weight_decay and self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        # Base optimizer
        base = self._create_optax_transform(self.learning_rate, **self.config)
        transforms.append(base)
        # Chain
        self.opt_transform = optax.chain(*transforms)

    @abstractmethod
    def _create_optax_transform(self, learning_rate: float, **config) -> optax.GradientTransformation:
        """
        Return an optax.GradientTransformation, e.g., optax.adam, optax.sgd.
        """
        pass

    def init(self, params):
        """
        Initialize optimizer state and optional accumulators.

        Args:
          params: PyTree of model parameters.

        Returns:
          (opt_state, accum_grads, ema_params, step)
        """
        opt_state = self.opt_transform.init(params)
        accum_grads = None
        if self.accumulate_steps and self.accumulate_steps > 1:
            accum_grads = jax.tree_map(jnp.zeros_like, params)
        ema_params = None
        if self.use_ema:
            ema_params = params
        step = 0
        return opt_state, accum_grads, ema_params, step

    def update(
        self,
        params,
        grads,
        opt_state,
        accum_grads,
        ema_params,
        step,
    ):
        """
        Pure-functionally apply gradients: compute new params, new opt_state,
        updated accum_grads and ema_params, and increment step.

        Returns:
          new_params, new_opt_state, new_accum_grads, new_ema_params, new_step, loss
        """
        # Loss scaling
        if self.loss_scale:
            grads = jax.tree_map(lambda g: g / self.loss_scale, grads)
        new_step = step + 1
        # Accumulation
        if self.accumulate_steps and self.accumulate_steps > 1:
            accum_grads = jax.tree_map(lambda a, g: a + g, accum_grads, grads)
            if new_step % self.accumulate_steps != 0:
                # no update yet; return same params
                return params, opt_state, accum_grads, ema_params, new_step, None
            # average
            grads = jax.tree_map(lambda a: a / self.accumulate_steps, accum_grads)
            # reset
            accum_grads = jax.tree_map(jnp.zeros_like, accum_grads)
        # Optax update
        updates, new_opt_state = self.opt_transform.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # EMA
        if self.use_ema:
            ema_params = jax.tree_map(
                lambda e, p: self.ema_decay * e + (1 - self.ema_decay) * p,
                ema_params, new_params
            )
            if self.ema_every and new_step % self.ema_every == 0:
                new_params = ema_params
        # Return
        return new_params, new_opt_state, accum_grads, ema_params, new_step, None

    def get_config(self):
        cfg = {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'clipnorm': self.clipnorm,
            'global_clipnorm': self.global_clipnorm,
            'loss_scale': self.loss_scale,
            'accumulate_steps': self.accumulate_steps,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay,
            'ema_every': self.ema_every,
        }
        cfg.update(self.config)
        return cfg
