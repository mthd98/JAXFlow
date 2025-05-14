import typing as tp
import abc
import random as pyrandom  # Import Python's random module
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Optional, Union

import jax.numpy as jnp
from jax import random
from jax.nn.initializers import (
    constant,
    delta_orthogonal,
    glorot_normal,
    glorot_uniform,
    he_normal,
    he_uniform,
    kaiming_normal,
    kaiming_uniform,
    lecun_normal,
    lecun_uniform,
    normal,
    ones,
    orthogonal,
    truncated_normal,
    uniform,
    variance_scaling,
    xavier_normal,
    xavier_uniform,
    zeros,
)


class Initializer(abc.ABC):
    """
    A base class for initializers.

    This class handles the common random key management, and
    forces subclasses to implement the __call__ method.
    Additionally, it stores a name for the initializer.
    """
    def __init__(self, seed: Optional[int] = 42, dtype=jnp.float32, name: str = None):
        # If seed is None, choose a random seed using Python's random module.
        if seed is None:
            seed = pyrandom.randint(0, 2**32 - 1)
        self.key = random.PRNGKey(seed)
        self.dtype = dtype
        self.name = name if name is not None else self.__class__.__name__

    @abc.abstractmethod
    def __call__(self, shape):
        """
        Initialize weights with the given shape and dtype.
        Must be implemented by subclasses.
        """
        pass

# Already implemented initializers:

class Constant(Initializer):
    def __init__(self, value: float, seed: int = 42, dtype=jnp.float32, name: str = None):
        super().__init__(seed=seed, dtype=dtype, name=name)
        self.value = value
        self.initializer = constant(self.value)

    def __call__(self, shape):
        return self.initializer(self.key, shape, self.dtype)

class DeltaOrthogonal(Initializer):
    def __init__(self, seed: int = 42, scale=1.0, column_axis=-1, dtype=jnp.float32, name: str = None):
        super().__init__(seed=seed, dtype=dtype, name=name)
        self.scale = scale
        self.column_axis = column_axis
        self.initializer = delta_orthogonal(self.scale, self.column_axis)

    def __call__(self, shape):
        return self.initializer(self.key, shape, self.dtype)

class GlorotNormal(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
        super().__init__(seed=seed, dtype=dtype, name=name)
        self.initializer = glorot_normal()

    def __call__(self, shape):
        return self.initializer(self.key, shape, self.dtype)

# Continuing with the rest of the initializers:

class GlorotUniform(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
        super().__init__(seed=seed, dtype=dtype, name=name)
        self.initializer = glorot_uniform()

    def __call__(self, shape):
        return self.initializer(self.key, shape, self.dtype)

class HeNormal(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
        super().__init__(seed=seed, dtype=dtype, name=name)
        self.initializer = he_normal()

    def __call__(self, shape):
        return self.initializer(self.key, shape, self.dtype)

class HeUniform(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.initializer = he_uniform()

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class KaimingNormal(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.initializer = kaiming_normal()

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class KaimingUniform(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.initializer = kaiming_uniform()

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class LecunNormal(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.initializer = lecun_normal()

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class LecunUniform(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.initializer = lecun_uniform()

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class Normal(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, mean: float = 0.0, stddev: float = 1.0, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.mean = mean
         self.stddev = stddev
         self.initializer = normal(stddev=self.stddev)

    def __call__(self, shape):
         # The 'normal' initializer returns samples with mean 0; we add our mean here.
         return self.initializer(self.key, shape, self.dtype) + self.mean

class Ones(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.initializer = ones

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class Orthogonal(Initializer):
    def __init__(self, seed: int = 42, gain: float = 1.0, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.gain = gain
         self.initializer = orthogonal(self.gain)

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class TruncatedNormal(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, stddev: float = 1.0, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.stddev = stddev
         self.initializer = truncated_normal(stddev=self.stddev)

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class Uniform(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, minval: float = 0.0, maxval: float = 1.0, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.minval = minval
         self.maxval = maxval
         self.initializer = uniform(scale=self.maxval - self.minval)

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class VarianceScaling(Initializer):
    def __init__(self, scale: float = 1.0, mode: str = 'fan_in', distribution: str = 'truncated_normal', seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.scale = scale
         self.mode = mode
         self.distribution = distribution
         self.initializer = variance_scaling(self.scale, self.mode, self.distribution)

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class XavierNormal(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.initializer = xavier_normal()

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class XavierUniform(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.initializer = xavier_uniform()

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)

class Zeros(Initializer):
    def __init__(self, seed: int = 42, dtype=jnp.float32, name: str = None):
         super().__init__(seed=seed, dtype=dtype, name=name)
         self.initializer = zeros

    def __call__(self, shape):
         return self.initializer(self.key, shape, self.dtype)
"""
# Example tests for the new initializers:
if __name__ == "__main__":
    # Dictionary mapping names to initializer instances (customize parameters as needed)
    init_funcs = {
        "Constant": Constant(value=3.0),
        "DeltaOrthogonal": DeltaOrthogonal(),
        "GlorotNormal": GlorotNormal(),
        "GlorotUniform": GlorotUniform(),
        "HeNormal": HeNormal(),
        "HeUniform": HeUniform(),
        "KaimingNormal": KaimingNormal(),
        "KaimingUniform": KaimingUniform(),
        "LecunNormal": LecunNormal(),
        "LecunUniform": LecunUniform(),
        "Normal": Normal(mean=0.0, stddev=1.0),
        "Ones": Ones(),
        "Orthogonal": Orthogonal(gain=1.0),
        "TruncatedNormal": TruncatedNormal(stddev=1.0),
        "Uniform": Uniform(minval=-1.0, maxval=1.0),
        "VarianceScaling": VarianceScaling(scale=2.0, mode='fan_out', distribution='uniform'),
        "XavierNormal": XavierNormal(),
        "XavierUniform": XavierUniform(),
        "Zeros": Zeros()
    }

    for name, initializer in init_funcs.items():
         weights = initializer((3, 3, 3))
         print(f"{initializer.name} initializer output:")
         print(weights)
         print("\n")
"""