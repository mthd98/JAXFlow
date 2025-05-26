import jax.numpy as jnp
from jaxflow.layers.layer import Layer

class GlobalAveragePooling2D(Layer):
    """
    Global average pooling 2D layer for NHWC tensors in JAXFlow.

    Computes the mean value of each feature map across the spatial dimensions
    (height and width) for every sample in the batch. This layer has no trainable
    parameters and works with both object-oriented and functional APIs.

    Args:
        keepdims (bool, optional): If True, retains singleton spatial dimensions,
            so the output shape is (batch, 1, 1, channels). Defaults to False,
            which produces shape (batch, channels).
        name (str or None, optional): Layer name. If None, a unique name is generated.
        trainable (bool, optional): Ignored (kept for API compatibility). Defaults to False.

    Inputs:
        inputs (jnp.ndarray): 4D tensor with shape (batch, height, width, channels).

    Input shape:
        (batch_size, height, width, channels)

    Output shape:
        (batch_size, channels) if keepdims=False (default);
        (batch_size, 1, 1, channels) if keepdims=True.

    Attributes:
        keepdims (bool): Whether to keep singleton spatial dimensions in output.
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax.numpy as jnp
        from jaxflow.layers.pooling import GlobalAveragePooling2D

        # Example input: batch of 2, height=3, width=5, 10 channels
        x = jnp.ones((2, 3, 5, 10))
        gap = GlobalAveragePooling2D()
        y = gap(x)
        print(y.shape)  # (2, 10)

        # With keepdims=True
        gap_keep = GlobalAveragePooling2D(keepdims=True)
        y2 = gap_keep(x)
        print(y2.shape)  # (2, 1, 1, 10)
        ```

    Raises:
        ValueError: If input is not a 4D tensor.

    Note:
        - No trainable parameters.
        - Reduces across height and width (axes 1 and 2).
        - Compatible with masking via the parent Layer's API.
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