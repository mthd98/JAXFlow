
import jax.numpy as jnp
from jaxflow.layers.layer import Layer

class Flatten(Layer):
    """Flatten layer for **jaxflow**.

    Transforms input tensors of shape *(batch, d1, d2, …, dn)* into
    *(batch, d1·d2·…·dn)* (i.e. flattens all spatial / temporal dims while
    keeping the leading batch dimension intact).

    This layer contains **no trainable parameters** – it simply reshapes the
    data.  Nonetheless it follows the same build / call conventions as other
    layers so it works seamlessly inside composite models and with the
    `functional_call` registry.
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


# test 
flatten = Flatten()
flatten.build((None, 3, 5))
x = jnp.ones((2, 3, 5))
flatten(x).shape

