import jax.numpy as jnp
from jax import lax
from jaxflow.layers.layer import Layer
from jaxflow.initializers.initializers import GlorotUniform, Zeros



class Dense(Layer):
    """Fully‑connected (affine) layer for **jaxflow**.

    Transforms the last dimension of the input tensor via a learned matrix and
    optional bias, followed by an optional non‑linear activation.

    ``y = activation(x @ W + b)``

    Parameters
    ----------
    units : int
        Number of output features (columns of *W*).
    activation : callable | None, default ``None``
        Activation function applied after the affine transform.  ``None`` means
        linear output.
    use_bias : bool, default ``True``
        Whether to include a bias term.
    kernel_initializer, bias_initializer : callable or Initializer subclass
        Objects that return an array given a ``shape`` argument.  Defaults are
        *GlorotUniform* for the kernel and *Zeros* for the bias.
    dtype : jnp.dtype, default ``jnp.float32``
        Data type of the parameters.
    device : str, default ``"auto"``
        Device placement for un‑sharded parameters (``"cpu"``, ``"gpu"``, …).
    shard_devices : list[jax.Device] | str | None
        If provided, parameters are sharded across these devices; ``"auto"``
        selects all devices matching *device*'s platform.
    seed : int | None
        Random seed forwarded to the initializers.
    name : str | None
        Layer name (defaults to class name).
    trainable : bool, default ``True``
        Whether the parameters participate in gradient updates.
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
