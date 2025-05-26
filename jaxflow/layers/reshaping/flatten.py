
import jax.numpy as jnp
from jaxflow.layers.layer import Layer

class Flatten(Layer):
    """
    Flatten layer for JAXFlow.

    Reshapes input tensors from (batch, d1, d2, ..., dn) to (batch, d1*d2*...*dn),
    flattening all dimensions except the leading batch axis. Contains no trainable
    parameters and is compatible with both object-oriented and functional APIs.

    Args:
        name (str, optional): Layer name. If None, a unique name is generated.
        trainable (bool, optional): Kept for API symmetry; always False.

    Inputs:
        inputs (jnp.ndarray): Tensor of shape (batch, d1, d2, ..., dn).

    Input shape:
        (batch_size, d1, d2, ..., dn)

    Output shape:
        (batch_size, d1*d2*...*dn)

    Attributes:
        built (bool): Whether the layer has been built.
        output_shape_ (tuple): Output shape after flattening.

    Example:
        ```python
        import jax.numpy as jnp
        from jaxflow.layers.flatten import Flatten

        # Example input: batch of 4, 8x8 feature maps
        x = jnp.ones((4, 8, 8))
        flatten = Flatten()
        y = flatten(x)
        print(y.shape)  # (4, 64)
        ```

    Raises:
        None

    Note:
        - No trainable parameters.
        - The layer reshapes the tensor but keeps the batch dimension intact.
        - Compatible with dynamic batch or sequence sizes (batch or spatial/temporal
          dimensions may be None at build time).
    """


    def __init__(self, *, name=None, trainable=False):
        # `trainable` is forced False but we keep the kwarg for API symmetry.
        super().__init__(name=name, trainable=trainable)

    # ------------------------------------------------------------------
    # Building – nothing to do except mark built & store input shape.
    # ------------------------------------------------------------------
    def build(self, input_shape):
        # Compute the total feature dimension after flattening (exclude batch).
        if None in input_shape[1:]:
            # We keep None for unknown dims (e.g. variable sequence length).
            flat_dim = None
        else:
            flat_dim = int(jnp.prod(jnp.array(input_shape[1:], dtype=jnp.int64)))
        self.output_shape_ = (input_shape[0], flat_dim)
        self.built = True
        self.built_shape = input_shape

    # ------------------------------------------------------------------
    # Forward pass (no parameters).
    # ------------------------------------------------------------------
    def call(self, inputs, **kwargs):  # kwargs keep training/mask signature
        # jnp.reshape with -1 flattens all remaining dims.
        return inputs.reshape((inputs.shape[0], -1))

    # Functional call – identical (no params to swap).
    def functional_call(self, inputs, params=None, **kwargs):
        return self.call(inputs, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        if not self.built:
            # Mirror logic from build.
            if None in input_shape[1:]:
                return (input_shape[0], None)
            flat_dim = int(jnp.prod(jnp.array(input_shape[1:], dtype=jnp.int64)))
            return (input_shape[0], flat_dim)
        return self.output_shape_

    def get_config(self):
        base = super().get_config()
        base.update({})  # No extra hyper‑parameters yet.
        return base

    def __repr__(self):
        return f"<Flatten built={self.built}, output_shape={getattr(self, 'output_shape_', None)}>"

