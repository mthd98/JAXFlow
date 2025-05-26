import jax.numpy as jnp
from jax import lax

from jaxflow.layers.layer import Layer



class MaxPooling2D(Layer):
    """
    Max pooling 2D layer for NHWC tensors in JAXFlow.

    Applies 2D max pooling over the height and width dimensions of 4D input tensors.
    Follows Keras semantics for pool_size and strides; supports both string padding
    ("VALID"/"SAME") and explicit tuple-based padding. No trainable parameters.

    Args:
        pool_size (int or tuple of int, optional): Size of the pooling window for each spatial dimension.
            If int, the same value is used for both height and width. Defaults to 2.
        strides (int or tuple of int or None, optional): Stride for the pooling window.
            If None, defaults to pool_size.
        padding (str or tuple, optional): Padding method: "VALID", "SAME", or
            explicit ((pad_h_lo, pad_h_hi), (pad_w_lo, pad_w_hi)). Defaults to "VALID".
        dilation (int or tuple of int, optional): Dilation rate for the pooling window.
            Rarely used, but supported for compatibility. Defaults to 1.
        keepdims (bool, optional): If True, retains singleton spatial dimensions in the output,
            so the output shape is (batch, 1, 1, channels) if the reduced dims are 1.
            Defaults to False.
        name (str or None, optional): Layer name. If None, a unique name is generated.

    Inputs:
        inputs (jnp.ndarray): 4D tensor of shape (batch, height, width, channels).

    Input shape:
        (batch_size, height, width, channels)

    Output shape:
        (batch_size, out_height, out_width, channels) if keepdims=False (default);
        (batch_size, 1, 1, channels) if keepdims=True and reduced dims are size 1.

    Attributes:
        pool_size (tuple): Size of the pooling window.
        strides (tuple): Stride for the pooling window.
        padding (str or tuple): Padding strategy.
        dilation (tuple): Dilation rate for the pooling window.
        keepdims (bool): Whether singleton spatial dims are kept in output.
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax.numpy as jnp
        from jaxflow.layers.pooling import MaxPooling2D

        # Example input: batch of 2, height=8, width=8, 3 channels
        x = jnp.arange(2 * 8 * 8 * 3).reshape(2, 8, 8, 3)
        pool = MaxPooling2D(pool_size=2, strides=2, padding="SAME")
        y = pool(x)
        print(y.shape)  # (2, 4, 4, 3)
        ```

    Raises:
        ValueError: If input is not a 4D tensor.

    Note:
        - No trainable parameters.
        - Pooling is performed over height and width (axes 1 and 2).
        - Supports both string and explicit tuple padding.
        - Keras/PyTorch-compatible API.
    """


    def __init__(
        self,
        pool_size=2,
        strides=None,
        *,
        padding="VALID",
        dilation=1,
        keepdims=False,
        name=None,
    ):
        super().__init__(name=name, trainable=False)

        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if strides is None:
            strides = pool_size
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.pool_size = tuple(pool_size)
        self.strides = tuple(strides)
        self.padding = padding  # str ("SAME"/"VALID") **or** explicit 2‑tuple
        self.dilation = tuple(dilation)
        self.keepdims = keepdims

    # ------------------------------------------------------------------
    # Build / call
    # ------------------------------------------------------------------

    def build(self, input_shape):  # noqa: D401
        if len(input_shape) != 4:
            raise ValueError(
                "MaxPooling2D expects NHWC inputs (batch, h, w, c). "
                f"Got shape {input_shape}."
            )
        self.built = True
        self.built_shape = input_shape

    # util helpers -----------------------------------------------------

    @staticmethod
    def _norm_tuple(x):
        if isinstance(x, int):
            return (x, x)
        if isinstance(x, tuple) and len(x) == 2:
            return x
        raise ValueError("Tuple must be int or length‑2 tuple")

    @staticmethod
    def _explicit_padding(pad):
        """Convert ((h_lo,h_hi),(w_lo,w_hi)) → full 4‑D padding for NHWC."""
        (ph_lo, ph_hi), (pw_lo, pw_hi) = pad
        return ((0, 0), (ph_lo, ph_hi), (pw_lo, pw_hi), (0, 0))

    # main forward -----------------------------------------------------

    def call(self, inputs, *, training=False, mask=None):  # noqa: D401
        window_dims = (1,) + self.pool_size + (1,)
        strides = (1,) + self.strides + (1,)
        dilation = (1,) + self.dilation + (1,)

        # Decide padding argument for lax.reduce_window
        if isinstance(self.padding, str):
            padding_arg = self.padding  # "SAME" / "VALID"
        else:
            padding_arg = self._explicit_padding(self.padding)

        out = lax.reduce_window(
            inputs,
            -jnp.inf,
            lax.max,
            window_dimensions=window_dims,
            window_strides=strides,
            padding=padding_arg,
            base_dilation=(1, 1, 1, 1),
            window_dilation=dilation,
        )

        if not self.keepdims and len(out.shape) == 4 and out.shape[1] == out.shape[2] == 1:
            out = out.reshape(out.shape[0], out.shape[-1])
        return out

    # functional variant ----------------------------------------------

    def functional_call(self, inputs, params=None, **kwargs):
        # No params – stateless – just forward to call()
        return self.call(inputs, **kwargs)

    # shape util -------------------------------------------------------

    def compute_output_shape(self, input_shape):
        n, h, w, c = input_shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        dh, dw = self.dilation

        if isinstance(self.padding, str) and self.padding == "SAME":
            oh = (h + sh - 1) // sh
            ow = (w + sw - 1) // sw
        else:  # VALID or explicit
            eff_ph = (ph - 1) * dh + 1
            eff_pw = (pw - 1) * dw + 1
            oh = (h - eff_ph) // sh + 1
            ow = (w - eff_pw) // sw + 1
        if self.keepdims:
            return (n, 1, 1, c)
        return (n, oh, ow, c)

    # misc -------------------------------------------------------------

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "pool_size": self.pool_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation": self.dilation,
                "keepdims": self.keepdims,
            }
        )
        return cfg

    def __repr__(self):
        return (
            f"<MaxPooling2D pool_size={self.pool_size}, strides={self.strides}, "
            f"padding={self.padding}, dilation={self.dilation}, built={self.built}>"
        )