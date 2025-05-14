import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jaxflow.layers.layer import Layer



class Dropout(Layer):
    """Inverted Dropout layer for **jaxflow**.

    At *training* time a random mask zeroes a fraction (*rate*) of the inputs
    and rescales the survivors by ``1 / (1‑rate)`` so the expected sum stays
    constant.  At inference the layer is the identity function.

    Parameters
    ----------
    rate : float in [0, 1)
        Probability of *dropping* (zeroing) each individual element.
    seed : int | None, default *None*
        Base RNG seed.  If *None* a random 32‑bit seed is chosen with NumPy.
    name : str | None
        Optional layer name.
    trainable : bool, default *False*
        Present for API consistency – Dropout has no parameters, so this flag
        has no effect.
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