import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jaxflow.layers.layer import Layer


class Dropout(Layer):
    """
    Inverted Dropout layer for **jaxflow** (JAX/Optax compatible).

    Randomly sets input units to zero with probability `rate` at training time,
    scaling the remaining activations by `1 / (1 - rate)` so the expected sum
    remains unchanged. No effect at inference (test) time.

    Supports both stateful (eager) and stateless (functional) APIs.

    Args:
        rate (float): Fraction of the input units to drop (between 0 and 1).
        seed (int, optional): Random seed for reproducibility. If None, uses a random seed.
        name (str, optional): Layer name.
        trainable (bool, optional): No effect (Dropout is stateless), for API symmetry.

    Example:
        >>> drop = Dropout(rate=0.5)
        >>> out = drop(x, training=True)

    Notes:
        - `call()` is not pure-functional (updates self._key); use `functional_call` in JIT/pmap or when using stateful training loops.
        - Dropout is only applied at training time (`training=True`).

    """

    def __init__(self, rate: float, *, seed: int | None = None, name=None, trainable=False):
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate must be in [0, 1). Got " + str(rate))
        super().__init__(name=name, trainable=trainable)

        self.rate = float(rate)
        self.scale = 1.0 / (1.0 - self.rate) if self.rate > 0.0 else 1.0
        self.seed = int(seed) if seed is not None else int(np.random.randint(0, 2**32 - 1))
        self._key = random.PRNGKey(self.seed)

    # ------------------------------------------------------------------
    # No variables to build – but keep the hook so .built flag behaves.
    # ------------------------------------------------------------------
    def build(self, input_shape):
        self.built = True
        self.built_shape = input_shape

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def call(self, inputs, *, training: bool = False, mask=None):  # noqa: D401
        if not training or self.rate == 0.0:
            return inputs  # identity at inference / when rate == 0

        # Split RNG for reproducibility under JIT.
        self._key, subkey = random.split(self._key)
        keep = random.bernoulli(subkey, p=1.0 - self.rate, shape=inputs.shape)
        return jnp.where(keep, inputs * self.scale, 0.0).astype(inputs.dtype)

    # ------------------------------------------------------------------
    # Pure functional variant (stateless)
    # ------------------------------------------------------------------
    def functional_call(self, inputs, params, *, rng_key, training: bool = False, mask=None):  # params unused
        if not training or self.rate == 0.0:
            return inputs, rng_key
        rng_key, sub = random.split(rng_key)
        keep = random.bernoulli(sub, p=1.0 - self.rate, shape=inputs.shape)
        return jnp.where(keep, inputs * self.scale, 0.0).astype(inputs.dtype), rng_key

    # ------------------------------------------------------------------
    # Mask propagation – Dropout does not alter spatial structure
    # ------------------------------------------------------------------
    def compute_mask(self, inputs, mask):
        return mask

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def get_config(self):
        base = super().get_config()
        base.update({"rate": self.rate, "seed": self.seed})
        return base

    def __repr__(self):
        return f"<Dropout rate={self.rate} built={self.built}>"