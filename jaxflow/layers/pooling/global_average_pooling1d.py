import jax.numpy as jnp
from jaxflow.layers.layer import Layer

class GlobalAveragePooling1D(Layer):
    """
    Global average pooling 1D layer for JAXFlow.

    Averages each feature map across the temporal/sequence (length) dimension
    for each sample in the batch. This layer has **no trainable parameters**.
    Supports object-oriented and functional APIs.

    Args:
        name (str, optional): Name for the layer. If None, a unique name is generated.
        keepdims (bool, optional): If True, retains a singleton length dimension
            (output shape (batch, 1, channels)). Defaults to False (output shape (batch, channels)).

    Inputs:
        inputs (jnp.ndarray): 3D tensor of shape (batch, length, channels).

    Input shape:
        (batch_size, length, channels)

    Output shape:
        (batch_size, channels) if keepdims=False (default);
        (batch_size, 1, channels) if keepdims=True.

    Attributes:
        keepdims (bool): Whether to keep the length dimension (as size 1).
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax.numpy as jnp
        from jaxflow.layers.pooling import GlobalAveragePooling1D

        # Example input: batch of 4, length 10, 8 channels
        x = jnp.ones((4, 10, 8))
        gap = GlobalAveragePooling1D()
        y = gap(x)
        print(y.shape)  # (4, 8)

        # With keepdims=True
        gap_keep = GlobalAveragePooling1D(keepdims=True)
        y2 = gap_keep(x)
        print(y2.shape)  # (4, 1, 8)
        ```

    Raises:
        ValueError: If input is not a 3D tensor.

    Note:
        - No trainable parameters.
        - Reduces across the sequence (axis=1).
        - Compatible with masking via the parent Layer's API.
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
