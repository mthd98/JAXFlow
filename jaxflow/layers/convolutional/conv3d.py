import jax.numpy as jnp
from jax import lax
from jaxflow.layers.layer import Layer
from jaxflow.initializers.initializers import GlorotUniform, Zeros

class Conv3D(Layer):
    """3‑D convolution layer for **jaxflow**.

    Expects inputs in **NDHWC** layout *(batch, depth, height, width, channels)*
    and produces outputs in the same layout.  Behaviour mirrors your `Conv2D`
    and `Conv1D` layers so you can swap dimensions with minimal changes.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
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
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides, strides)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.filters = filters
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides)
        self.padding = padding
        self.dilation = tuple(dilation)
        self.groups = groups

        # Default initializers
        if kernel_initializer is None:
            kernel_initializer = GlorotUniform
        if bias_initializer is None:
            bias_initializer = Zeros

        self.kernel_initializer = (
            kernel_initializer(seed=seed, dtype=dtype)
            if callable(kernel_initializer)
            else kernel_initializer
        )
        self.bias_initializer = (
            bias_initializer(seed=seed, dtype=dtype)
            if callable(bias_initializer)
            else bias_initializer
        )

        # Store any extra kwargs (currently unused)
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

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def build(self, input_shape):
        """Create Variables once the input shape is known."""
        if len(input_shape) != 5:
            raise ValueError(
                "Conv3D expects NDHWC inputs (batch, depth, height, width, channels). "
                f"Got shape {input_shape}."
            )
        in_channels = input_shape[-1]
        if in_channels % self.groups != 0:
            raise ValueError(
                "in_channels must be divisible by groups "
                f"(got in_channels={in_channels}, groups={self.groups})."
            )
        kd, kh, kw = self.kernel_size
        weight_shape = (kd, kh, kw, in_channels // self.groups, self.filters)
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

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(self, inputs, training=False, mask=None):
        if inputs.shape[-1] != (self.kernel.shape[3] * self.groups):
            raise ValueError(
                "Last input dim (channels) does not match kernel shape. "
                f"inputs.channels={inputs.shape[-1]}, kernel.in_channels={self.kernel.shape[3]} × groups={self.groups}."
            )
        outputs = lax.conv_general_dilated(
            lhs=inputs,
            rhs=self.kernel.value,  # unwrap Variable → ndarray
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=(1, 1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
            feature_group_count=self.groups,
        )
        if self.use_bias:
            outputs = outputs + self.bias.value  # broadcast over D/H/W
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    # ------------------------------------------------------------------
    # Functional (pure) variant
    # ------------------------------------------------------------------
    def functional_call(self, inputs, params, training=False, mask=None):
        kernel = params["kernel"]
        bias = params.get("bias", None)
        outputs = lax.conv_general_dilated(
            lhs=inputs,
            rhs=kernel,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=(1, 1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
            feature_group_count=self.groups,
        )
        if bias is not None:
            outputs = outputs + bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        """Return the layer's output shape for SAME/VALID padding."""
        n, d, h, w, _ = input_shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.strides
        dd, dh, dw = self.dilation
        if self.padding == "SAME":
            od = (d + sd - 1) // sd
            oh = (h + sh - 1) // sh
            ow = (w + sw - 1) // sw
        else:  # VALID
            od = (d - (kd - 1) * dd - 1) // sd + 1
            oh = (h - (kh - 1) * dh - 1) // sh + 1
            ow = (w - (kw - 1) * dw - 1) // sw + 1
        return (n, od, oh, ow, self.filters)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
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
        return cfg

    def __repr__(self):
        cfg = self.get_config()
        return (
            f"<Conv3D filters={cfg['filters']}, kernel_size={cfg['kernel_size']}, "
            f"strides={cfg['strides']}, padding={cfg['padding']}, groups={cfg['groups']}, built={self.built}>"
        )



