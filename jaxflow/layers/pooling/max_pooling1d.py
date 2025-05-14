import jax.numpy as jnp
from jax import lax

from jaxflow.layers.layer import Layer





class MaxPooling1D(Layer):
    """Max‑Pooling layer for **1‑D** feature maps *(N, L, C)*.

    *pool_size* and *strides* follow Keras semantics; padding may be
    "VALID"/"SAME" (upper‑case) **or** an explicit tuple ``((pad_left, pad_right),)``.
    """

    def __init__(
        self,
        pool_size: int,
        strides: int | None = None,
        *,
        padding: str | tuple[tuple[int, int]] = "VALID",
        dilation: int = 1,
        keepdims: bool = False,
        name: str | None = None,
        trainable: bool = False,
    ):
        super().__init__(name=name, trainable=trainable)
        self.pool_size = int(pool_size)
        self.strides = int(strides) if strides is not None else self.pool_size
        self.padding = padding  # str or explicit tuple
        self.dilation = int(dilation)
        self.keepdims = keepdims

    # ------------------------------------------------------------------
    # Build is a no‑op – the layer has no parameters
    # ------------------------------------------------------------------
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "MaxPooling1D expects input shape (batch, length, channels); "
                f"got {input_shape}.")
        self.built = True
        self.built_shape = input_shape

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _explicit_padding(self):
        """Return explicit padding tuple for lax when *self.padding* is a tuple."""
        # self.padding is ((left, right),) for the length axis → insert batch & channel
        return ((0, 0),) + self.padding + ((0, 0),)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(self, inputs, training=False, mask=None):  # noqa: D401
        window_dims = (self.pool_size,)
        strides = (self.strides,)
        dilation = (self.dilation,)

        if isinstance(self.padding, str):
            pad_arg = self.padding.upper()
            out = lax.reduce_window(
                inputs,
                -jnp.inf,
                lax.max,
                window_dimensions=(1,) + window_dims + (1,),
                window_strides=(1,) + strides + (1,),
                padding=pad_arg,
                base_dilation=(1, 1, 1),
                window_dilation=(1,) + dilation + (1,),
            )
        else:
            pad_arg = self._explicit_padding()
            out = lax.reduce_window(
                inputs,
                -jnp.inf,
                lax.max,
                window_dimensions=(1,) + window_dims + (1,),
                window_strides=(1,) + strides + (1,),
                padding=pad_arg,
                base_dilation=(1, 1, 1),
                window_dilation=(1,) + dilation + (1,),
            )

        # Remove length dim if keepdims=False and it is 1
        if not self.keepdims and out.shape[1] == 1:
            out = jnp.squeeze(out, axis=1)
        return out

    # ------------------------------------------------------------------
    # Functional variant
    # ------------------------------------------------------------------
    def functional_call(self, inputs, params, training=False, mask=None):
        # No parameters – reuse call
        return self.call(inputs, training=training, mask=mask)

    # ------------------------------------------------------------------
    # Shapes & config helpers
    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        n, l, c = input_shape
        if isinstance(self.padding, str) and self.padding.upper() == "SAME":
            out_l = (l + self.strides - 1) // self.strides
        elif isinstance(self.padding, str):  # VALID
            out_l = (l - (self.pool_size - 1) * self.dilation - 1) // self.strides + 1
        else:  # explicit
            pad_total = self.padding[0][0] + self.padding[0][1]
            out_l = (l + pad_total - (self.pool_size - 1) * self.dilation - 1) // self.strides + 1
        if self.keepdims:
            return (n, 1, c)
        return (n, out_l, c)

    def get_config(self):
        base = super().get_config()
        base.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
            "dilation": self.dilation,
            "keepdims": self.keepdims,
        })
        return base

    def __repr__(self):
        cfg = self.get_config()
        return (
            f"<MaxPooling1D pool_size={cfg['pool_size']}, strides={cfg['strides']}, "
            f"padding={cfg['padding']}, dilation={cfg['dilation']}, built={self.built}>" )
