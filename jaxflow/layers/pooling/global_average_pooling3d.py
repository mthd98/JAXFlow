import jax.numpy as jnp
from jaxflow.layers.layer import Layer

class GlobalAveragePooling3D(Layer):
    """Global Average Pooling **3‑D** layer for *jaxflow*.

    *Input shape*  : ``(batch, depth, height, width, channels)``  (N, D, H, W, C)

    *Output shape* :
      * ``(batch, channels)``  if ``keepdims=False`` (default)
      * ``(batch, 1, 1, 1, channels)``  if ``keepdims=True``

    The layer averages each feature map over the three spatial dimensions
    (D×H×W).  It contains **no trainable parameters** but adheres to the full
    *jaxflow* `Layer` API so it can sit anywhere in a model.

    Args:
        keepdims (bool, optional): If *True*, retain singleton spatial dims.
        name (str, optional): Layer name.  Defaults to class name.
        trainable (bool, optional): Kept for API symmetry (ignored).
    """

    def __init__(self, *, keepdims: bool = False, name: str | None = None, trainable: bool = True):
        super().__init__(name=name, trainable=trainable)
        self.keepdims = bool(keepdims)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def build(self, input_shape):  # noqa: D401 – matches Layer signature
        if len(input_shape) != 5:
            raise ValueError(
                "GlobalAveragePooling3D expects NDHWC inputs (batch, depth, height, width, channels). "
                f"Got shape {input_shape}."
            )
        # Nothing to build (no variables) – just mark as built.
        self.built = True
        self.built_shape = input_shape

    def call(self, inputs, training=False, mask=None):  # noqa: D401
        # Mean over D, H, W  ➜ axes (1, 2, 3)
        return jnp.mean(inputs, axis=(1, 2, 3), keepdims=self.keepdims)

    # ------------------------------------------------------------------
    # Optional functional variant (trivial – no params)
    # ------------------------------------------------------------------
    def functional_call(self, inputs, params, training=False, mask=None):
        return jnp.mean(inputs, axis=(1, 2, 3), keepdims=self.keepdims)

    # ------------------------------------------------------------------
    # Mask handling (pass-through)
    # ------------------------------------------------------------------
    def compute_mask(self, inputs, mask):  # noqa: D401
        return mask

    # ------------------------------------------------------------------
    # Shape helpers & config
    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        if self.keepdims:
            return (input_shape[0], 1, 1, 1, input_shape[-1])
        else:
            return (input_shape[0], input_shape[-1])

    def get_config(self):
        base = super().get_config()
        base.update({"keepdims": self.keepdims})
        return base

    def __repr__(self):
        return (
            f"<GlobalAveragePooling3D keepdims={self.keepdims}, built={self.built}>"
        )
