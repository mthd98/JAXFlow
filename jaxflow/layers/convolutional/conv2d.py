import jax.numpy as jnp
from jax import lax
from jaxflow.initializers.initializers import GlorotUniform, Zeros
from jaxflow.layers.layer import Layer

class Conv2D(Layer):
    """
    2D convolution layer for image and spatial data in JAXFlow.

    This layer creates a trainable 2D convolution kernel and applies it to input tensors
    of shape (batch, height, width, channels) using grouped, dilated, or standard convolution.
    It supports lazy variable creation, device placement, sharding, and is compatible with
    both object-oriented and functional APIs.

    Args:
        filters (int): Number of output feature maps (channels).
        kernel_size (int or tuple of int): Height and width of the convolution kernel.
            If an int, uses the same value for both dimensions.
        strides (int or tuple of int, optional): Stride for the spatial dimensions.
            Defaults to 1.
        padding (str or tuple, optional): "SAME", "VALID", or explicit padding. Defaults to "SAME".
        dilation (int or tuple of int, optional): Dilation rate for dilated convolution. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        activation (callable, optional): Activation function to apply after bias. Defaults to None (linear).
        use_bias (bool, optional): Whether to add a learnable bias. Defaults to True.
        kernel_initializer (callable, optional): Initializer for the kernel weights.
            Defaults to GlorotUniform.
        bias_initializer (callable, optional): Initializer for the bias. Defaults to Zeros.
        device (str, optional): Device for parameter placement ("auto", "cpu", "gpu", "tpu"). Defaults to "auto".
        shard_devices (list or str, optional): Devices for parameter sharding. See Variable docs.
        dtype (jax.numpy.dtype, optional): Data type for parameters. Defaults to float32.
        trainable (bool, optional): Whether the layer is trainable. Defaults to True.
        name (str, optional): Name for the layer. If None, a unique name is generated.
        seed (int, optional): Random seed for parameter initialization.
        kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint (callable, optional):
            Placeholders for Keras compatibility (stored but not used).

    Inputs:
        inputs (jnp.ndarray): 4D tensor of shape (batch, height, width, channels).

    Input shape:
        (batch_size, height, width, in_channels)

    Output shape:
        (batch_size, new_height, new_width, filters)
        where new_height and new_width depend on padding, kernel_size, strides, and dilation.

    Attributes:
        filters (int): Number of output channels.
        kernel_size (tuple): (height, width) of the convolution kernel.
        strides (tuple): Strides along each spatial dimension.
        padding (str or tuple): Padding strategy.
        dilation (tuple): Dilation rate.
        groups (int): Number of groups for grouped convolution.
        use_bias (bool): Whether a bias is included.
        kernel (Variable): The kernel variable.
        bias (Variable): The bias variable, if use_bias is True.
        activation (callable): The activation function.
        device (str): Device for kernel/bias placement.
        shard_devices (list or str): Devices for sharding.
        dtype (jax.numpy.dtype): Data type of the kernel and bias.
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from jaxflow.layers.conv2d import Conv2D

        # Example input: batch of 8, 32x32 RGB images (3 channels)
        x = jnp.ones((8, 32, 32, 3))
        # Conv2D layer: 16 filters, 3x3 kernel, stride 2, ReLU activation
        conv = Conv2D(16, (3, 3), strides=2, activation=jax.nn.relu)
        y = conv(x)
        print(y.shape)  # (8, 16, 16, 16) if padding="SAME"
        ```

    Raises:
        ValueError: If input shape, group/channel configuration, or variable shapes are invalid.

    Note:
        - Input must have 4 dimensions: (batch, height, width, channels) (NHWC layout).
        - For grouped convolution, in_channels must be divisible by groups.
        - Output shape depends on padding, kernel size, strides, and dilation.
        - Kernel and bias variables are created lazily during the first call.
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
