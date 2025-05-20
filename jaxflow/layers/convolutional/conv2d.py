import jax.numpy as jnp
from jax import lax
from jaxflow.initializers.initializers import GlorotUniform, Zeros
from jaxflow.layers.layer import Layer

class Conv2D(Layer):
    """2‑D convolution layer for *jaxflow*.

    Args:
        filters (int): Number of output feature maps.
        kernel_size (int | tuple[int, int]): Height and width of the convolution
            kernel. If an int is given, the same value is used for both dims.
        strides (int | tuple[int, int], optional): Stride for the spatial
            dimensions. Defaults to 1.
        padding (str | tuple[tuple[int, int], tuple[int, int]], optional):
            ``"SAME"`` or ``"VALID"`` (upper‑case, as expected by ``jax.lax``)
            or an explicit padding configuration. Defaults to ``"SAME"``.
        dilation (int | tuple[int, int], optional): Dilation rate for atrous
            convolution. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution.
            ``groups==in_channels`` gives a depth‑wise convolution. Defaults to 1.
        activation (callable, optional): Activation applied after bias. ``None``
            yields a linear layer (no activation). Defaults to ``None``.
        use_bias (bool, optional): Whether to learn an additive bias.
            Defaults to ``True``.
        kernel_initializer (callable, optional): Instance of an initializer
            class **or** a callable that returns an array given ``shape``. The
            constructor must accept ``seed`` and ``dtype`` kwargs. Defaults to
            :class:`~jaxflow.initializers.GlorotUniform`.
        bias_initializer (callable, optional): Initializer for bias. Defaults to
            :class:`~jaxflow.initializers.Zeros`.
        device (str, optional): ``"auto"``, ``"cpu"``, ``"gpu"`` or ``"tpu"``.
        shard_devices (list | str | None, optional): Devices for parameter
            sharding. Passed through to :class:`~jaxflow.core.variable.Variable`.
        dtype (jax.numpy.dtype, optional): Parameter dtype. Defaults to
            ``jnp.float32``.
        seed (int | None, optional): Base random seed. If ``None``, a random
            seed is chosen.
        **regularizer_and_constraint_kwargs:  Place‑holders for regularizer /
            constraint callables so the signature is drop‑in compatible with
            Keras‑style layers – these kwargs are stored but otherwise ignored
            for now so they don't break user code.
    """

    def __init__(
        self,
        filters: int,
        kernel_size,
        strides=1,
        padding="SAME",
        dilation=1,
        groups: int = 1,
        *,
        name=None,
        device="auto",
        shard_devices=None,
        dtype=jnp.float32,
        trainable=True,
        activation=None,
        use_bias: bool = True,
        kernel_initializer=None,
        bias_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.filters = filters
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides)
        self.padding = padding
        self.dilation = tuple(dilation)
        self.groups = groups

        # Handle initializers – allow strings, classes, or callables.

        if kernel_initializer is None:
            kernel_initializer = GlorotUniform
        if bias_initializer is None:
            bias_initializer = Zeros

        # Instantiate the initializer objects so we can call them in build().
        self.kernel_initializer = kernel_initializer(seed=seed, dtype=dtype) if callable(kernel_initializer) else kernel_initializer
        self.bias_initializer = bias_initializer(seed=seed, dtype=dtype) if callable(bias_initializer) else bias_initializer

        # Store misc kwargs even if unused
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.activation = activation
        self.use_bias = use_bias
        self.device = device
        self.shard_devices = shard_devices
        self.dtype = dtype
        self.seed = seed

    # ---------------------------------------------------------------------
    # Building + forward pass
    # ---------------------------------------------------------------------

    def build(self, input_shape):
        """Create *Variable* parameters lazily once the input shape is known."""

        if len(input_shape) != 4:
            raise ValueError(
                "Conv2D expects NHWC inputs (batch, height, width, channels). "
                f"Got shape {input_shape}."
            )
        in_channels = input_shape[-1]
        if in_channels % self.groups != 0:
            raise ValueError(
                "in_channels must be divisible by groups "
                f"(got in_channels={in_channels}, groups={self.groups})."
            )

        kh, kw = self.kernel_size
        weight_shape = (kh, kw, in_channels // self.groups, self.filters)
        kernel_init_val = self.kernel_initializer(shape=weight_shape)

        self.kernel = self.add_variable(
            name="kernel",
            initial_value=kernel_init_val,
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )

        if self.use_bias:
            bias_init_val = self.bias_initializer(shape=(self.filters,))
            self.bias = self.add_variable(
                name="bias",
                initial_value=bias_init_val,
                device=self.device,
                shard_devices=self.shard_devices,
                dtype=self.dtype,
                trainable=self.trainable,
            )

    def call(self, inputs, training=False, mask=None):  # noqa: D401 – we match base signature
        if inputs.shape[-1] != (self.kernel.shape[2] * self.groups):
            raise ValueError(
                "Last input dim (channels) does not match kernel shape. "
                f"inputs.channels={inputs.shape[-1]}, kernel.in_channels={self.kernel.shape[2]} × groups={self.groups}."
            )

        # lax expects (NHWC, HWIO, NHWC) dimension numbers if kernel shape is HWIO
        outputs = lax.conv_general_dilated(
            lhs=inputs,
            rhs=self.kernel.value,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups,
        )

        if self.use_bias:
            outputs = outputs + self.bias  # broadcast over spatial dims

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
        

    # ---------------------------------------------------------------------
    # Optional functional pass (re‑wire to *kernel* & *bias* provided)
    # ---------------------------------------------------------------------

    def functional_call(self, inputs, params, training=False, mask=None):
        """Pure functional variant – use *params* instead of layer variables."""
        kernel = params["kernel"]
        bias = params.get("bias", None)

        outputs = lax.conv_general_dilated(
            lhs=inputs,
            rhs=kernel,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups,
        )

        if bias is not None:
            outputs = outputs + bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation": self.dilation,
                "groups": self.groups,
                "use_bias": self.use_bias,
            }
        )
        return base

    def __repr__(self):
        cfg = self.get_config()
        return (
            f"<Conv2D filters={cfg['filters']}, kernel_size={cfg['kernel_size']}, "
            f"strides={cfg['strides']}, padding={cfg['padding']}, "
            f"groups={cfg['groups']}, built={self.built}>"
        )
