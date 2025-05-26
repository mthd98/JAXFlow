"""
metrics.py

Base Metric class for jaxflow, following a simplified Keras-like API pattern.
"""

from typing import Any, List, Optional, Dict
import jax.numpy as jnp

from jaxflow.core.variable import Variable
from jaxflow.initializers import Zeros
from jaxflow.core.auto_name import AutoNameMixin

class Metric(AutoNameMixin):
    """
    Base Metric class for JAXFlow.

    Implements a lightweight, Keras-inspired API for stateful or stateless metrics
    in JAXFlow. Subclass this and override `update_state()` and `result()` to
    create custom metrics.

    Args:
        name (str, optional): Name for the metric instance. If None, a unique name is auto-generated.
        dtype (jnp.dtype, optional): Data type used for computations and variables. Defaults to jnp.float32.

    Inputs:
        The expected inputs to `update_state()` and `__call__()` depend on the specific metric subclass.
        Most metrics expect `y_true`, `y_pred`, and optionally `sample_weight` or other arguments.

    Attributes:
        name (str): Name for the metric instance.
        dtype (jnp.dtype): Data type for all computations.
        variables (list[Variable]): List of all internal state variables used to accumulate statistics.

    Methods:
        update_state(...): Accumulate statistics for the metric. Must be implemented by subclasses.
        result(): Compute and return the current metric value. Must be implemented by subclasses.
        reset_state(): Reset all metric state variables to zero.
        stateless_update_state(metric_vars, ...): Functional/stateless update (see Notes).
        stateless_result(metric_vars): Functional/stateless result computation (see Notes).
        get_config(): Return a serializable config dictionary for this metric.
        from_config(config): Re-create the metric from a config dict.

    Usage Example:
        ```python
        class MeanMetric(Metric):
            def __init__(self, name=None, dtype=None):
                super().__init__(name=name, dtype=dtype)
                self.total = self.add_variable("total", shape=())
                self.count = self.add_variable("count", shape=())

            def update_state(self, values):
                self.total.assign(self.total.value + jnp.sum(values))
                self.count.assign(self.count.value + values.size)

            def result(self):
                return self.total.value / (self.count.value + 1e-7)

        metric = MeanMetric()
        metric.update_state(jnp.array([1, 2, 3]))
        print(metric.result())  # Output: 2.0
        metric.reset_state()
        ```

    Notes:
        - All metric state is stored in non-trainable `Variable` instances.
        - Metrics can be used statefully (default) or functionally/statelessly via
          `stateless_update_state` and `stateless_result`.
        - Subclasses **must** call `super().__init__()` and implement both
          `update_state()` and `result()`.
        - `reset_state()` sets all internal state variables to zero.
        - The `variables` property returns a list of all stateful `Variable` objects.
        - Metrics are compatible with JAX transformations (jit/vmap/pmap) via stateless helpers.

    Raises:
        RuntimeError: If `super().__init__()` was not called in a subclass.
        NotImplementedError: If `update_state()` or `result()` are not implemented in a subclass.
        ValueError: For mismatched variable array lengths in stateless helpers.

    See Also:
        - Keras Metric base class: https://keras.io/api/metrics/
        - JAXFlow Variable: `jaxflow.core.variable.Variable`
    """

    def __init__(self, name: Optional[str] = None, dtype: Any = None):
        # Required attributes
        self.name = self.auto_name(name)
        self.dtype = dtype or jnp.float32
        self._variables: List[Variable] = []
        # Marker to verify super init call
        self._initialized = True

    def _check_super_called(self):
        if not hasattr(self, '_initialized'):
            raise RuntimeError(
                "You forgot to call super().__init__() in the Metric subclass __init__."
            )

    def __setattr__(self, name: str, value: Any):
        # Allow normal assignment for initialization marker
        super().__setattr__(name, value)

    def add_variable(
        self,
        name: str,
        shape: tuple = (),
        initializer: Any = None,
        dtype: Any = None
    ) -> Variable:
        """
        Create and track a new non-trainable state variable.
        """
        self._check_super_called()
        init = initializer or Zeros
        var = Variable(
           initial_value=init(shape),
            dtype=dtype or self.dtype,
            trainable=False,
            name=name
        )
        self._variables.append(var)
        return var

    @property
    def variables(self) -> List[Variable]:
        """All state variables."""
        return list(self._variables)

    def reset_state(self):
        """
        Reset all variables to zero.
        """
        self._check_super_called()
        for v in self._variables:
            v.assign(jnp.zeros(v.shape, dtype=v.dtype))

    def update_state(self, *args, **kwargs):
        """
        Accumulate statistics. Must be implemented by subclasses.
        """
        self._check_super_called()
        raise NotImplementedError("Must implement update_state in subclass.")

    def result(self):
        """
        Compute and return metric value. Must be implemented by subclasses.
        """
        self._check_super_called()
        raise NotImplementedError("Must implement result in subclass.")

    def stateless_update_state(
        self,
        metric_vars: List[jnp.ndarray],
        *args,
        **kwargs
    ) -> List[jnp.ndarray]:
        """
        Stateless update: given a list of variable arrays, returns updated variable arrays
        without mutating original state.
        """
        self._check_super_called()
        if len(metric_vars) != len(self._variables):
            raise ValueError(
                f"Expected {len(self._variables)} variables, got {len(metric_vars)}"
            )
        # Save current state
        old_vals = [v.value for v in self._variables]
        # Assign temporary state
        for v, val in zip(self._variables, metric_vars):
            v.assign(val)
        # Perform update
        self.update_state(*args, **kwargs)
        # Collect new state
        new_vals = [v.value for v in self._variables]
        # Restore
        for v, val in zip(self._variables, old_vals):
            v.assign(val)
        return new_vals

    def stateless_result(
        self,
        metric_vars: List[jnp.ndarray]
    ) -> Any:
        """
        Stateless result: compute metric result given variable arrays,
        without mutating original state.
        """
        self._check_super_called()
        if len(metric_vars) != len(self._variables):
            raise ValueError(
                f"Expected {len(self._variables)} variables, got {len(metric_vars)}"
            )
        old_vals = [v.value for v in self._variables]
        for v, val in zip(self._variables, metric_vars):
            v.assign(val)
        res = self.result()
        for v, val in zip(self._variables, old_vals):
            v.assign(val)
        return res

    def __call__(self, *args, **kwargs) -> Any:
        """
        Shortcut: update_state then result.
        """
        self._check_super_called()
        self.update_state(*args, **kwargs)
        return self.result()

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable config of the metric."""
        self._check_super_called()
        return {"name": self.name, "dtype": self.dtype}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Metric":
        return cls(**config)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"

    def __str__(self) -> str:
        return self.__repr__()
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Metric):
            return NotImplemented
        return self.name == other.name and self.dtype == other.dtype

    def __hash__(self) -> int:
        return hash((self.name, self.dtype))


    def __len__(self) -> int:
        """
        Return the number of state variables.
        """
        return len(self._variables)
