import jax.numpy as jnp
from jax import lax
from jaxflow.layers.layer import Layer
from jaxflow.initializers.initializers import GlorotUniform, Zeros

class Conv1D(Layer):
    """
    1D convolution layer for temporal data in JAXFlow.

    This layer applies a 1-dimensional convolution over an input signal
    composed of several input channels. It supports grouped convolutions,
    lazy variable creation, flexible device placement, and is compatible 
    with pure-functional and object-oriented APIs.

    The expected input data format is `(batch, length, channels)` (NWC).
    Convolution is performed over the 'length' axis. Output shape and device
    placement are handled automatically. API is similar to Keras and 
    TensorFlow's `Conv1D`, but optimized for JAX workflows.

    Args:
        filters (int): Number of output channels.
        kernel_size (int): Length of the 1D convolution window.
        strides (int, optional): Stride of the convolution. Defaults to 1.
        padding (str, optional): One of 'SAME' or 'VALID'. Defaults to 'SAME'.
        dilation (int, optional): Dilation rate for dilated convolution. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. 
            `in_channels` must be divisible by `groups`. Defaults to 1.
        name (str, optional): Name for the layer. If None, a unique name is generated.
        device (str, optional): Device for kernel/bias placement ('auto', 'cpu', 'gpu', 'tpu'). Defaults to 'auto'.
        shard_devices (list or str, optional): Devices for sharding the kernel/bias. See Variable docs.
        dtype (jax.numpy.dtype, optional): Data type of the variables. Defaults to float32.
        trainable (bool, optional): Whether the layer's parameters are trainable. Defaults to True.
        activation (callable, optional): Activation function to use. If None, no activation is applied.
        use_bias (bool, optional): Whether to add a bias. Defaults to True.
        kernel_initializer (callable, optional): Initializer for the convolution kernel.
            Defaults to `GlorotUniform`.
        bias_initializer (callable, optional): Initializer for the bias vector.
            Defaults to `Zeros`.
        kernel_regularizer (callable, optional): Optional kernel regularizer.
        bias_regularizer (callable, optional): Optional bias regularizer.
        activity_regularizer (callable, optional): Optional activity regularizer.
        kernel_constraint (callable, optional): Optional kernel constraint.
        bias_constraint (callable, optional): Optional bias constraint.
        seed (int, optional): Random seed for initialization.

    Inputs:
        inputs (jnp.ndarray): 3D tensor of shape `(batch_size, length, channels)`.

    Input shape:
        (batch_size, length, in_channels)

    Output shape:
        (batch_size, new_length, filters)
        where `new_length` depends on `padding`, `kernel_size`, `strides`, and `dilation`.

    Attributes:
        filters (int): Number of output channels.
        kernel_size (int): Length of the 1D convolution window.
        strides (int): Stride of the convolution.
        padding (str): Padding method.
        dilation (int): Dilation rate.
        groups (int): Number of groups for grouped convolution.
        use_bias (bool): Whether a bias is used.
        kernel (Variable): Convolution kernel variable.
        bias (Variable): Bias variable, if `use_bias` is True.
        device (str): Device string for placement.
        shard_devices (list or str): Devices for sharding.
        dtype (jax.numpy.dtype): Data type of the kernel and bias.
        activation (callable): Activation function.
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from jaxflow.layers.conv1d import Conv1D

        # Example input: batch of 4, sequence length 100, 16 channels
        x = jnp.ones((4, 100, 16))
        # Create Conv1D layer: 32 output filters, kernel size 3, stride 2, ReLU activation
        conv = Conv1D(32, 3, strides=2, activation=jax.nn.relu)
        y = conv(x)  # triggers build, runs convolution
        print(y.shape)  # (4, 50, 32)
        ```
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        *,
        strides: int = 1,
        padding: str = "SAME",
        dilation: int = 1,
        groups: int = 1,
        name: str | None = None,
        device: str = "auto",
        shard_devices=None,
        dtype=jnp.float32,
        trainable: bool = True,
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

        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = padding
        self.dilation = int(dilation)
        self.groups = int(groups)

        if kernel_initializer is None:
            kernel_initializer = GlorotUniform
        if bias_initializer is None:
            bias_initializer = Zeros

        self.kernel_initializer = (
            kernel_initializer(seed=seed, dtype=dtype) if callable(kernel_initializer) else kernel_initializer
        )
        self.bias_initializer = (
            bias_initializer(seed=seed, dtype=dtype) if callable(bias_initializer) else bias_initializer
        )

        self.activation = activation
        self.use_bias = use_bias
        self.device = device
        self.shard_devices = shard_devices
        self.dtype = dtype
        self.seed = seed

        # keep for API parity even if unused yet
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    # ------------------------------------------------------------------
    # Build + forward
    # ------------------------------------------------------------------

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Conv1D expects NLC inputs (batch, length, channels). "
                f"Got shape {input_shape}."
            )
        in_channels = input_shape[-1]
        if in_channels % self.groups != 0:
            raise ValueError(
                "in_channels must be divisible by groups "
                f"(got {in_channels} vs groups={self.groups})."
            )

        weight_shape = (self.kernel_size, in_channels // self.groups, self.filters)
        kernel_val = self.kernel_initializer(shape=weight_shape)
        self.kernel = self.add_variable(
            name="kernel",
            initial_value=kernel_val,
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )

        if self.use_bias:
            bias_val = self.bias_initializer(shape=(self.filters,))
            self.bias = self.add_variable(
                name="bias",
                initial_value=bias_val,
                device=self.device,
                shard_devices=self.shard_devices,
                dtype=self.dtype,
                trainable=self.trainable,
            )

    def call(self, inputs, training=False, mask=None):
        outputs = lax.conv_general_dilated(
            lhs=inputs,
            rhs=self.kernel.value,            # WIO (kernel, in, out)
            window_strides=(self.strides,),
            padding=self.padding,
            lhs_dilation=(1,),
            rhs_dilation=(self.dilation,),
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=self.groups,
        )
        if self.use_bias:
            outputs = outputs + self.bias.value  # broadcast over length
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    # Functional variant (pure, jit‑friendly)
    def functional_call(self, inputs, params, training=False, mask=None):
        kernel = params["kernel"]
        bias = params.get("bias", None)
        outputs = lax.conv_general_dilated(
            lhs=inputs,
            rhs=kernel,
            window_strides=(self.strides,),
            padding=self.padding,
            lhs_dilation=(1,),
            rhs_dilation=(self.dilation,),
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=self.groups,
        )
        if bias is not None:
            outputs = outputs + bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    # Utility -----------------------------------------------------------
    def compute_output_shape(self, input_shape):
        n, l, _ = input_shape
        if self.padding == "SAME":
            out_len = (l + self.strides - 1) // self.strides
        else:  # VALID
            dil_k = (self.kernel_size - 1) * self.dilation + 1
            out_len = (l - dil_k + self.strides) // self.strides
        return (n, out_len, self.filters)

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
            f"<Conv1D filters={cfg['filters']}, kernel_size={cfg['kernel_size']}, "
            f"strides={cfg['strides']}, padding={cfg['padding']}, groups={cfg['groups']}, "
            f"built={self.built}>"
        )


# ----------------------------------------------------------------------
# Example quick‑test
# ----------------------------------------------------------------------
"""if __name__ == "__main__":
    import jax

    x = jnp.ones((4, 100, 16))                # batch=4, length=100, channels=16
    conv = Conv1D(32, 3, strides=2, activation=jax.nn.relu)
    y = conv(x)                               # triggers build
    print("output shape", y.shape)            # (4, 50, 32)
    print(conv)
"""

