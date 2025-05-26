import jax.numpy as jnp
from jaxflow.layers.layer import Layer
from jaxflow.initializers import GlorotUniform, Zeros,Ones

class LayerNormalization(Layer):
    """
    Layer normalization layer for JAXFlow (Ba et al., 2016).

    Normalizes activations across the specified feature axes for each sample independently,
    stabilizing hidden-state dynamics and improving training speed. Supports centering
    and scaling via trainable parameters. API and behavior closely match Keras and PyTorch.

    Args:
        axis (int or tuple of int): Axis or axes that should be normalized (typically the features axis).
            Negative values are supported and refer to axes from the end.
        epsilon (float, optional): Small constant added to variance to avoid division by zero.
            Defaults to 1e-5.
        center (bool, optional): If True, add offset (beta) to normalized tensor. Defaults to True.
        scale (bool, optional): If True, multiply by scale (gamma). Defaults to True.
        beta_initializer (callable or Initializer, optional): Initializer for beta. Defaults to Zeros.
        gamma_initializer (callable or Initializer, optional): Initializer for gamma. Defaults to Ones.
        name (str, optional): Layer name. If None, a unique name is generated.
        device (str, optional): Device for parameter placement ("auto", "cpu", "gpu", "tpu"). Defaults to "auto".
        shard_devices (list or str, optional): Devices for parameter sharding. See Variable docs.
        dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
        trainable (bool, optional): Whether parameters are trainable. Defaults to True.

    Inputs:
        inputs (jnp.ndarray): Arbitrary-rank tensor. Normalization is performed over `axis`.

    Input shape:
        Any shape. Most commonly (batch_size, features) or (batch_size, ..., features).

    Output shape:
        Same as input shape.

    Attributes:
        axis (tuple): Axes being normalized.
        epsilon (float): Epsilon added to variance.
        center (bool): Whether to learn beta (offset).
        scale (bool): Whether to learn gamma (scale).
        beta (Variable): Beta parameter (if center=True).
        gamma (Variable): Gamma parameter (if scale=True).
        dtype (jnp.dtype): Data type of parameters.
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from jaxflow.layers.normalization import LayerNormalization

        # Example input: batch of 16, 64 features
        x = jnp.ones((16, 64))
        ln = LayerNormalization(axis=-1)
        y = ln(x)
        print(y.shape)  # (16, 64)
        ```

    Raises:
        ValueError: If input shape does not match required axes.

    Note:
        - Supports centering (beta) and scaling (gamma) via trainable parameters.
        - Axis can be negative or a tuple of axes.
        - Output always has the same shape as input.
        - Compatible with both object-oriented and functional APIs.
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


