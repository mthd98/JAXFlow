import jax.numpy as jnp
from jaxflow.layers.layer import Layer

class GlobalAveragePooling3D(Layer):
    """
    Global average pooling 3D layer for NDHWC tensors in JAXFlow.

    Averages each feature map over the three spatial dimensions (depth, height, width)
    for every sample in the batch. This layer contains **no trainable parameters** and 
    fits seamlessly into any JAXFlow model.

    Args:
        keepdims (bool, optional): If True, retains singleton spatial dimensions, 
            so the output shape is (batch, 1, 1, 1, channels). Defaults to False,
            which produces output shape (batch, channels).
        name (str, optional): Layer name. If None, a unique name is generated.
        trainable (bool, optional): Kept for API compatibility (ignored). Defaults to True.

    Inputs:
        inputs (jnp.ndarray): 5D tensor of shape (batch, depth, height, width, channels).

    Input shape:
        (batch_size, depth, height, width, channels)

    Output shape:
        (batch_size, channels) if keepdims=False (default);
        (batch_size, 1, 1, 1, channels) if keepdims=True.

    Attributes:
        keepdims (bool): Whether to keep singleton spatial dimensions in the output.
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax.numpy as jnp
        from jaxflow.layers.pooling import GlobalAveragePooling3D

        # Example input: batch of 2, depth=4, height=5, width=6, 3 channels
        x = jnp.ones((2, 4, 5, 6, 3))
        gap = GlobalAveragePooling3D()
        y = gap(x)
        print(y.shape)  # (2, 3)

        # With keepdims=True
        gap_keep = GlobalAveragePooling3D(keepdims=True)
        y2 = gap_keep(x)
        print(y2.shape)  # (2, 1, 1, 1, 3)
        ```

    Raises:
        ValueError: If input is not a 5D tensor.

    Note:
        - No trainable parameters.
        - Reduces across the three spatial dimensions (axes 1, 2, 3).
        - Compatible with masking via the parent Layer's API.
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
