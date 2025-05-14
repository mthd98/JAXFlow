import jax.numpy as jnp
from jaxflow.layers.layer import Layer
from jaxflow.initializers import GlorotUniform, Zeros,Ones

class LayerNormalization(Layer):
    """Layer Normalization layer for **jaxflow** (Ba et al., 2016).

    Normalizes across the **feature axes** of each sample independently,
    stabilising hidden‑state dynamics and improving training speed.

    Parameters follow the conventions of Keras / PyTorch so you can swap code
    easily.
    """

    def __init__(
        self,
        axis=-1,
        *,
        epsilon: float = 1e-5,
        center: bool = True,
        scale: bool = True,
        beta_initializer=None,
        gamma_initializer=None,
        name=None,
        device="auto",
        shard_devices=None,
        dtype=jnp.float32,
        trainable: bool = True,
    ):
        super().__init__(name=name, trainable=trainable)

        # Accept int or tuple/list and canonicalise to positive tuple.
        if isinstance(axis, int):
            axis = (axis,)
        axis = tuple(axis)
        self.axis = axis

        self.epsilon = epsilon
        self.center = center
        self.scale = scale

        # Default initialisers
        beta_initializer = beta_initializer or Zeros
        gamma_initializer = gamma_initializer or Ones
        self.beta_initializer = beta_initializer(dtype=dtype)
        self.gamma_initializer = gamma_initializer(dtype=dtype)

        self.device = device
        self.shard_devices = shard_devices
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, input_shape):
        # Compute param shape as the dims along *axis*
        param_shape = tuple(input_shape[a] for a in self.axis)

        if self.center:
            beta_val = self.beta_initializer(shape=param_shape)
            self.beta = self.add_variable(
                "beta",
                initial_value=beta_val,
                device=self.device,
                shard_devices=self.shard_devices,
                dtype=self.dtype,
                trainable=self.trainable,
            )
        if self.scale:
            gamma_val = self.gamma_initializer(shape=param_shape)
            self.gamma = self.add_variable(
                "gamma",
                initial_value=gamma_val,
                device=self.device,
                shard_devices=self.shard_devices,
                dtype=self.dtype,
                trainable=self.trainable,
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _normalize(self, x, beta=None, gamma=None):
        # Compute mean & variance over the specified axes keeping dims for broadcast.
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        var = jnp.var(x, axis=self.axis, keepdims=True)
        inv = jnp.reciprocal(jnp.sqrt(var + self.epsilon))
        y = (x - mean) * inv
        if gamma is not None:
            y = y * gamma
        if beta is not None:
            y = y + beta
        return y

    def call(self, inputs, training=False, mask=None):
        beta = getattr(self, "beta", None)
        gamma = getattr(self, "gamma", None)
        beta = beta if beta is not None else None
        gamma = gamma.value if gamma is not None else None
        return self._normalize(inputs, beta, gamma)

    # ------------------------------------------------------------------
    # Functional variant
    # ------------------------------------------------------------------
    def functional_call(self, inputs, params, training=False, mask=None):
        beta = params.get("beta", None)
        gamma = params.get("gamma", None)
        return self._normalize(inputs, beta, gamma)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                axis=self.axis,
                epsilon=self.epsilon,
                center=self.center,
                scale=self.scale,
            )
        )
        return cfg

    def __repr__(self):
        return (
            f"<LayerNorm axis={self.axis}, epsilon={self.epsilon}, "
            f"center={self.center}, scale={self.scale}, built={self.built}>"
        )


