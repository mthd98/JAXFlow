import jax.numpy as jnp
from jax import lax
from jaxflow.layers.layer import Layer
from jaxflow.initializers.initializers import GlorotUniform, Zeros

class Dense(Layer):
    """
    Fully-connected (affine) layer for JAXFlow.

    Transforms the last dimension of the input tensor via a trainable matrix
    and optional bias, followed by an optional activation function:
    `output = activation(inputs @ kernel + bias)`

    Args:
        units (int): Number of output features (the last dimension of the output).
        activation (callable, optional): Activation function applied after the affine transform.
            If None, no activation is applied (linear output). Defaults to None.
        use_bias (bool, optional): Whether to include a bias term. Defaults to True.
        kernel_initializer (callable or Initializer, optional): Initializer for the kernel weights.
            Defaults to GlorotUniform.
        bias_initializer (callable or Initializer, optional): Initializer for the bias. Defaults to Zeros.
        dtype (jnp.dtype, optional): Data type for the parameters. Defaults to jnp.float32.
        device (str, optional): Device for parameter placement ("auto", "cpu", "gpu", "tpu"). Defaults to "auto".
        shard_devices (list or str, optional): Devices for sharding parameters. See Variable docs.
        seed (int, optional): Random seed for initializers.
        name (str, optional): Layer name. If None, a unique name is generated.
        trainable (bool, optional): Whether the parameters are trainable. Defaults to True.

    Inputs:
        inputs (jnp.ndarray): Tensor of rank >= 2. The last dimension must match the kernel input.

    Input shape:
        (..., input_dim), where input_dim is the size of the last dimension.

    Output shape:
        (..., units), replacing the last dimension with `units`.

    Attributes:
        units (int): Number of output features.
        activation (callable or None): Activation function.
        use_bias (bool): Whether bias is used.
        kernel (Variable): Kernel (weight matrix) variable.
        bias (Variable): Bias variable, if use_bias is True.
        device (str): Device for parameter placement.
        dtype (jnp.dtype): Data type of the parameters.
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from jaxflow.layers.dense import Dense

        # Example input: batch of 16, 64 features
        x = jnp.ones((16, 64))
        dense = Dense(32, activation=jax.nn.relu)
        y = dense(x)
        print(y.shape)  # (16, 32)
        ```

    Raises:
        ValueError: If input shape does not match kernel dimensions.

    Note:
        - Inputs must have rank >= 2 (usually [batch_size, input_dim]).
        - Last dimension of inputs must match kernel input dimension.
        - Supports device placement and sharding for distributed training.
    """


    def __init__(
        self,
        units: int,
        *,
        activation=None,
        use_bias: bool = True,
        kernel_initializer=None,
        bias_initializer=None,
        dtype=jnp.float32,
        device="auto",
        shard_devices=None,
        seed=None,
        name=None,
        trainable=True,
    ):
        super().__init__(name=name, trainable=trainable)

        self.units = int(units)
        self.activation = activation
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.device = device
        self.shard_devices = shard_devices
        self.seed = seed

        # Instantiate initializers (defaults if None)
        if kernel_initializer is None:
            kernel_initializer = GlorotUniform
        if bias_initializer is None:
            bias_initializer = Zeros

        self.kernel_initializer = kernel_initializer(seed=seed, dtype=dtype) if callable(kernel_initializer) else kernel_initializer
        self.bias_initializer = bias_initializer(seed=seed, dtype=dtype) if callable(bias_initializer) else bias_initializer

    # ------------------------------------------------------------------
    # Build & forward
    # ------------------------------------------------------------------

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError("Dense expects inputs with rank ≥2 (batch + features).")
        in_features = input_shape[-1]

        kernel_shape = (in_features, self.units)
        kernel_init_val = self.kernel_initializer(shape=kernel_shape)

        self.kernel = self.add_variable(
            name="kernel",
            initial_value=kernel_init_val,
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )

        if self.use_bias:
            bias_init_val = self.bias_initializer(shape=(self.units,))
            self.bias = self.add_variable(
                name="bias",
                initial_value=bias_init_val,
                device=self.device,
                shard_devices=self.shard_devices,
                dtype=self.dtype,
                trainable=self.trainable,
            )

    def call(self, inputs, training=False, mask=None):  # noqa: D401 – keep signature
        if inputs.shape[-1] != self.kernel.shape[0]:
            raise ValueError(
                f"Last input dim ({inputs.shape[-1]}) must match kernel's first dim ({self.kernel.shape[0]})."
            )

        y = jnp.matmul(inputs, self.kernel.value)  # (…, in) @ (in, out) → (…, out)
        if self.use_bias:
            y = y + self.bias.value
        if self.activation is not None:
            y = self.activation(y)
        return y

    # ------------------------------------------------------------------
    # Functional variant
    # ------------------------------------------------------------------

    def functional_call(self, inputs, params, training=False, mask=None):
        y = jnp.matmul(inputs, params["kernel"])
        bias = params.get("bias")
        if bias is not None:
            y = y + bias
        if self.activation is not None:
            y = self.activation(y)
        return y

    # ------------------------------------------------------------------
    # Helpers / metadata
    # ------------------------------------------------------------------

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "units": self.units,
                "activation": getattr(self.activation, "__name__", str(self.activation)),
                "use_bias": self.use_bias,
            }
        )
        return base

    def __repr__(self):
        cfg = self.get_config()
        return (
            f"<Dense units={cfg['units']}, activation={cfg['activation']}, "
            f"built={self.built}>"
        )
