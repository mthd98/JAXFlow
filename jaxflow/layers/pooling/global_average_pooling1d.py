import jax.numpy as jnp
from jaxflow.layers.layer import Layer

class GlobalAveragePooling1D(Layer):
    """Global Average Pooling **1‑D** layer for *jaxflow*.

    *Input shape*  : ``(batch, length, channels)`` (N, L, C)

    *Output shape* : ``(batch, channels)`` if ``keepdims=False`` *(default)*,
    else ``(batch, 1, channels)``.

    This layer has **no trainable parameters** – it simply averages each
    feature map over the temporal/sequence *length* dimension.
    """

    def __init__(self, *, name=None, keepdims: bool = False):
        super().__init__(name=name, trainable=False)
        self.keepdims = keepdims

    # ------------------------------------------------------------
    # Layer lifecycle
    # ------------------------------------------------------------
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "GlobalAveragePooling1D expects (batch, length, channels) input; "
                f"got {input_shape}"
            )
        self.built = True
        self.built_shape = input_shape

    def call(self, inputs, training=False, mask=None):  # noqa: D401 – keep signature
        return jnp.mean(inputs, axis=1, keepdims=self.keepdims)

    # Functional alias (no params)
    def functional_call(self, inputs, params=None, training=False, mask=None):
        return self.call(inputs, training=training, mask=mask)

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        batch, _, channels = input_shape
        if self.keepdims:
            return (batch, 1, channels)
        return (batch, channels)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"keepdims": self.keepdims})
        return cfg

    def __repr__(self):
        return (
            f"<GlobalAveragePooling1D keepdims={self.keepdims}, built={self.built}>"
        )
