import optax
import jax
import jax.numpy as jnp

class BaseOptimizer:
    """
    An improved base optimizer for jaxflow that wraps optax and includes:
      - Gradient clipping (global or per-variable)
      - Weight decay
      - Loss scaling for mixed precision
      - Gradient accumulation
      - Exponential moving average (EMA) tracking

    Subclasses should override `_create_optax_optimizer` to provide the
    core gradient transformation (e.g., for Adam or SGD).
    """
    def __init__(
        self,
        learning_rate,
        weight_decay=0.0,
        clipnorm=None,
        global_clipnorm=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        use_ema=False,
        ema_decay=0.99,
        ema_apply_frequency=None,
        **kwargs,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.global_clipnorm = global_clipnorm
        self.loss_scale_factor = loss_scale_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_apply_frequency = ema_apply_frequency
        self.config = kwargs

        # Build the underlying optax transformation chain.
        self._build_optimizer()
        self.opt_state = None
        self.accumulated_grads = None  # For gradient accumulation.
        self.step_count = 0
        self.ema_params = None  # For EMA tracking.

    def _build_optimizer(self):
        # Get the core transformation from the subclass.
        base_transformation = self._create_optax_optimizer(self.learning_rate, **self.config)
        transformations = []

        # Apply gradient clipping.
        if self.global_clipnorm is not None:
            transformations.append(optax.clip_by_global_norm(self.global_clipnorm))
        elif self.clipnorm is not None:
            # Note: optax.clip clips each element to [-clipnorm, clipnorm].
            transformations.append(optax.clip(self.clipnorm))

        # Apply weight decay.
        if self.weight_decay:
            transformations.append(optax.add_decayed_weights(self.weight_decay))

        # Append the core transformation (e.g., Adam or SGD update rule).
        transformations.append(base_transformation)

        # Chain all the transformations together.
        self.opt_transform = optax.chain(*transformations)

    def _create_optax_optimizer(self, learning_rate, **kwargs):
        """
        Subclasses should implement this method to return an optax.GradientTransformation.
        For example, an Adam optimizer might return:
            optax.adam(learning_rate=learning_rate, b1=..., b2=..., eps=...)
        """
        raise NotImplementedError("Subclasses must implement _create_optax_optimizer")

    def init(self, params):
        """
        Initialize the optimizer state given the model parameters.

        Args:
            params: A PyTree of model parameters.

        Returns:
            The initial optimizer state.
        """
        self.opt_state = self.opt_transform.init(params)
        if self.gradient_accumulation_steps is not None:
            # Create a zero PyTree matching the structure of params.
            self.accumulated_grads = jax.tree_map(jnp.zeros_like, params)
        if self.use_ema:
            # Initialize EMA parameters as a copy of the initial parameters.
            self.ema_params = params
        self.step_count = 0
        return self.opt_state

    def apply_gradients(self, params, grads):
        """
        Update the model parameters using the provided gradients.

        Args:
            params: A PyTree of model parameters.
            grads: A PyTree of gradients (with same structure as params).

        Returns:
            Updated model parameters.
        """
        # Adjust gradients if loss scaling is used.
        if self.loss_scale_factor:
            grads = jax.tree_map(lambda g: g / self.loss_scale_factor, grads)

        # If using gradient accumulation, accumulate and update only every N steps.
        if self.gradient_accumulation_steps is not None:
            # Accumulate gradients.
            self.accumulated_grads = jax.tree_map(lambda acc, g: acc + g,
                                                  self.accumulated_grads, grads)
            self.step_count += 1
            # Only update parameters when the accumulation counter reaches the threshold.
            if self.step_count % self.gradient_accumulation_steps != 0:
                return params
            # Average the accumulated gradients.
            grads = jax.tree_map(lambda g: g / self.gradient_accumulation_steps, self.accumulated_grads)
            # Reset accumulated gradients.
            self.accumulated_grads = jax.tree_map(jnp.zeros_like, self.accumulated_grads)

        # Compute updates and new optimizer state.
        updates, new_state = self.opt_transform.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        self.opt_state = new_state

        # Update EMA if enabled.
        if self.use_ema:
            self.ema_params = jax.tree_map(
                lambda ema, p: self.ema_decay * ema + (1 - self.ema_decay) * p,
                self.ema_params, new_params
            )
            # Optionally, overwrite parameters with EMA values every specified steps.
            if self.ema_apply_frequency and (self.step_count % self.ema_apply_frequency == 0):
                new_params = self.ema_params

        return new_params

    def get_config(self):
        """
        Return the configuration of the optimizer for serialization.
        """
        config = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "clipnorm": self.clipnorm,
            "global_clipnorm": self.global_clipnorm,
            "loss_scale_factor": self.loss_scale_factor,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "ema_apply_frequency": self.ema_apply_frequency,
        }
        config.update(self.config)
        return config
