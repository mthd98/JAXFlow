import jax.numpy as jnp
from jax import lax

from jaxflow.layers.layer import Layer



class MaxPooling2D(Layer):
    """Max‑Pooling layer for 2‑D feature maps (**NHWC** tensors).

    Args
    -----
    pool_size : int | tuple[int, int]
        Window height/width. If an int is given the same value is used for both
        spatial dims.
    strides : int | tuple[int, int] | None
        Stride for the pooling window. If ``None`` it defaults to ``pool_size``.
    padding : str | tuple[tuple[int, int], tuple[int, int]]
        ``"VALID"`` or ``"SAME"`` (upper‑case, as expected by ``jax.lax``), or
        an explicit padding configuration ``((pad_h_lo, pad_h_hi), (pad_w_lo, pad_w_hi))``.
    dilation : int | tuple[int, int]
        Dilation of the pooling window.  Rarely used but kept for parity with
        Keras / PyTorch. Defaults to ``1``.
    keepdims : bool
        If *True*, retains singleton spatial dims in the output, yielding
        ``(N, 1, 1, C)`` instead of ``(N, C)`` when the input height/width equal
        the pool size.  (Default: *False* – no extra dims.)
    name : str | None
        Layer name.
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