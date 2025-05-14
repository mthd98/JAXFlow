import jax.numpy as jnp
from jaxflow.layers.layer import Layer

class GlobalAveragePooling2D(Layer):
    """Global Average Pooling layer for **NHWC** tensors.

    Input shape  : *(batch, height, width, channels)*
    Output shape : *(batch, channels)*

    **Behaviour** – computes the mean of each feature map over the spatial
    dimensions (H×W).  No trainable parameters.

    Args:
        keepdims (bool): If *True*, retains singleton spatial dims so the output
            shape is *(batch, 1, 1, channels)*.  Defaults to *False*.
        name (str | None): Layer name.
        trainable (bool): Ignored (kept for API parity).
    """

    def __init__(self, *, keepdims: bool = False, name: str | None = None, trainable: bool = False):
        super().__init__(name=name, trainable=trainable)
        self.keepdims = keepdims

    # ------------------------------------------------------------------
    # Build – nothing to build but set flags for completeness
    # ------------------------------------------------------------------
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("GlobalAveragePooling2D expects NHWC inputs (batch, H, W, C). "
                             f"Got shape {input_shape}.")
        # no variables to create

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(self, inputs, training=False, mask=None):  # noqa: D401
        # Mean over height & width axes (1,2)
        outputs = jnp.mean(inputs, axis=(1, 2), keepdims=self.keepdims)
        return outputs

    # ------------------------------------------------------------------
    # Functional variant (identical – stateless)
    # ------------------------------------------------------------------
    def functional_call(self, inputs, params=None, training=False, mask=None):
        return jnp.mean(inputs, axis=(1, 2), keepdims=self.keepdims)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        if self.keepdims:
            return (input_shape[0], 1, 1, input_shape[3])
        return (input_shape[0], input_shape[3])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"keepdims": self.keepdims})
        return cfg

    def __repr__(self):
        return f"<GlobalAveragePooling2D keepdims={self.keepdims}, built={self.built}>"


# test 
gap = GlobalAveragePooling2D(keepdims=False)
gap.build((None, 3, 5, 10))
x = jnp.ones((2, 3, 5, 10))
gap(x).shape