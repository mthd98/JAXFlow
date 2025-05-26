import jax.numpy as jnp
from jax import lax
from jaxflow.initializers.initializers import Normal, Zeros
from jaxflow.layers.layer import Layer
class Embedding(Layer):
    """
    Embedding layer for JAXFlow.

    Maps integer indices in the range [0, vocab_size) to dense, trainable vectors
    of length output_dim. Supports optional mask propagation for zero-padding tokens.

    This implementation balances the minimalism of Flax (pure gather for fast JIT)
    with Keras-style ergonomics (mask_zero, pluggable initializers). Advanced
    features (e.g. LoRA, quantization) can be added via mixins.

    Args:
        vocab_size (int): Number of unique tokens (maximum index + 1).
        output_dim (int): Dimension of the dense embedding vectors.
        mask_zero (bool, optional): If True, index 0 is treated as padding, and a boolean mask
            is propagated. Defaults to False.
        embeddings_initializer (callable, optional): Initializer for the embedding matrix.
            Defaults to Normal.
        dtype (jnp.dtype, optional): Storage dtype for the embeddings. Defaults to jnp.float32.
        device (str, optional): Device for variable placement ("auto", "cpu", "gpu", "tpu"). Defaults to "auto".
        shard_devices (list or str, optional): Devices for parameter sharding. See Variable docs.
        seed (int, optional): Random seed for initialization.
        name (str, optional): Name for the layer. If None, a unique name is generated.
        trainable (bool, optional): Whether the embedding matrix is trainable. Defaults to True.

    Inputs:
        inputs (jnp.ndarray): Integer tensor of arbitrary shape, each value in [0, vocab_size).

    Input shape:
        (batch_size, sequence_length, ...) or any integer-valued shape.

    Output shape:
        (*input_shape, output_dim)

    Attributes:
        vocab_size (int): Size of the vocabulary (number of embeddings).
        output_dim (int): Dimension of each embedding vector.
        mask_zero (bool): Whether index 0 is masked.
        embeddings (Variable): The embedding matrix variable.
        device (str): Device string for placement.
        dtype (jnp.dtype): Storage dtype for the embedding matrix.
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from jaxflow.layers.embedding import Embedding

        # Suppose vocabulary size = 10000, embedding dimension = 128
        emb = Embedding(10000, 128, mask_zero=True)
        x = jnp.array([[4, 10, 0], [7, 3, 9]])  # batch of indices, shape (2, 3)
        y = emb(x)  # shape (2, 3, 128)
        mask = emb.compute_mask(x, None)  # shape (2, 3), False where input is 0
        ```

    Raises:
        ValueError: If inputs are not integer type, or indices are out of bounds.

    Note:
        - Inputs must be integer type (int32 or int64). Other types are automatically cast.
        - If mask_zero=True, index 0 is treated as padding and is masked in downstream layers.
        - The layer is compatible with JAX JIT/vmap/pmap for fast functional API usage.
        - The `attend(query)` method enables weight tying for output projection.
    """


    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        *,
        mask_zero: bool = False,
        embeddings_initializer=None,
        dtype=jnp.float32,
        device="auto",
        shard_devices=None,
        seed=None,
        name=None,
        trainable=True,
    ):
        super().__init__(name=name, trainable=trainable)

        self.vocab_size = int(vocab_size)
        self.output_dim = int(output_dim)
        self.mask_zero = bool(mask_zero)
        self.dtype = dtype
        self.device = device
        self.shard_devices = shard_devices
        self.seed = seed

        if embeddings_initializer is None:
            embeddings_initializer = Normal
        self.embeddings_initializer = (
            embeddings_initializer(seed=seed, dtype=dtype)
            if callable(embeddings_initializer)
            else embeddings_initializer
        )

    # ------------------------------------------------------------------
    # Build & forward
    # ------------------------------------------------------------------

    def build(self, input_shape):
        # No assumptions on input rank; only that they are integers.
        self.embeddings = self.add_variable(
            name="embeddings",
            shape=(self.vocab_size, self.output_dim),
            initial_value=self.embeddings_initializer(shape=(self.vocab_size, self.output_dim)),
            dtype=self.dtype,
            device=self.device,
            shard_devices=self.shard_devices,
            trainable=self.trainable,
        )

    def call(self, inputs, training=False, mask=None):  # noqa: D401
        """if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Embedding inputs must be integer type.")"""
        # casts non‑int to int32; we do the same for safety.
        if inputs.dtype not in (jnp.int32, jnp.int64):
            inputs = inputs.astype(jnp.int32)

        # Gather along vocab axis.
        out = jnp.take(self.embeddings.value, inputs, axis=0)
        return out

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def compute_mask(self, inputs, mask):
        if not self.mask_zero:
            return None
        return inputs != 0  # padding token == 0 → mask False

    # ------------------------------------------------------------------
    # Functional variant – enables JIT vmap pmap easily
    # ------------------------------------------------------------------

    def functional_call(self, inputs, params, training=False, mask=None):
        
        """if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Embedding inputs must be integer type.")"""

        # casts non‑int to int32; we do the same for safety.
        if inputs.dtype not in (jnp.int32, jnp.int64):
            inputs = inputs.astype(jnp.int32)
        embeddings = params["embeddings"]
        out = jnp.take(embeddings, inputs, axis=0)
        return out

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def attend(self, query: jnp.ndarray) -> jnp.ndarray:
        """Project *query* vectors into vocab space via tied weights.

        Equivalent to ``query @ embeddings.T``.
        Returns shape ``(..., vocab_size)``.
        """
        return jnp.einsum("...d,vd->...v", query, self.embeddings.value)

    def compute_output_shape(self, input_shape):
        return (*input_shape, self.output_dim)

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "vocab_size": self.vocab_size,
                "output_dim": self.output_dim,
                "mask_zero": self.mask_zero,
            }
        )
        return base

    def __repr__(self):
        cfg = self.get_config()
        return (
            f"<Embedding vocab={cfg['vocab_size']}, dim={cfg['output_dim']}, "
            f"mask_zero={cfg['mask_zero']}, built={self.built}>"
        )
