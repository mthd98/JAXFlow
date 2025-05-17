import jax.numpy as jnp
from typing import Optional, Any, Dict

from jaxflow.core.variable import Variable
from jaxflow.initializers import Zeros
from jaxflow.metrics.metric import Metric

# --- Functional metrics ---

def mse(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Mean Squared Error: average of (y_pred - y_true)^2, optionally weighted.
    """
    y_true = jnp.asarray(y_true, dtype=jnp.float32)
    y_pred = jnp.asarray(y_pred, dtype=jnp.float32)
    err = y_pred - y_true
    sq = err**2
    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32).ravel()
        w = jnp.broadcast_to(w[:, None], sq.shape) if sq.ndim>1 else w
        return jnp.sum(sq * w) / (jnp.sum(w) + 1e-8)
    return jnp.mean(sq)


def mae(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Mean Absolute Error: average of |y_pred - y_true|, optionally weighted.
    """
    y_true = jnp.asarray(y_true, dtype=jnp.float32)
    y_pred = jnp.asarray(y_pred, dtype=jnp.float32)
    err = jnp.abs(y_pred - y_true)
    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32).ravel()
        w = jnp.broadcast_to(w[:, None], err.shape) if err.ndim>1 else w
        return jnp.sum(err * w) / (jnp.sum(w) + 1e-8)
    return jnp.mean(err)


def rmse(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Root Mean Squared Error: sqrt of MSE.
    """
    return jnp.sqrt(mse(y_true, y_pred, sample_weight))


def r2_score(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Coefficient of determination R^2 = 1 - SS_res / SS_tot
    Supports optional weighting.
    """
    y_true = jnp.asarray(y_true, dtype=jnp.float32)
    y_pred = jnp.asarray(y_pred, dtype=jnp.float32)
    err = y_true - y_pred
    ss_res = err**2
    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32).ravel()
        w = jnp.broadcast_to(w[:, None], ss_res.shape) if ss_res.ndim>1 else w
        ss_res = jnp.sum(ss_res * w)
        y_mean = jnp.sum(y_true * w) / (jnp.sum(w) + epsilon)
        ss_tot = jnp.sum(((y_true - y_mean)**2) * w)
    else:
        ss_res = jnp.sum(ss_res)
        y_mean = jnp.mean(y_true)
        ss_tot = jnp.sum((y_true - y_mean)**2)
    return 1 - ss_res / (ss_tot + epsilon)

# --- Functional API ---


# --- Stateful Metric classes ---

class MeanSquaredError(Metric):
    """Stateful Mean Squared Error."""
    def __init__(self, name: str = 'mse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_sq = self.add_variable('sum_sq', (), initializer=Zeros(), dtype=self.dtype)
        self.sum_w  = self.add_variable('sum_w',  (), initializer=Zeros(), dtype=self.dtype)

    def update_state(self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None):
        y_true = jnp.asarray(y_true, dtype=self.dtype)
        y_pred = jnp.asarray(y_pred, dtype=self.dtype)
        sq = (y_pred - y_true)**2
        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype).ravel()
            w = jnp.broadcast_to(w[:, None], sq.shape) if sq.ndim>1 else w
            sum_sq = jnp.sum(sq * w)
            sum_w  = jnp.sum(w)
        else:
            sum_sq = jnp.sum(sq)
            sum_w  = sq.size
        self.sum_sq.assign(self.sum_sq + sum_sq)
        self.sum_w.assign(self.sum_w + sum_w)

    def result(self) -> jnp.ndarray:
        return self.sum_sq.value / (self.sum_w.value + 1e-8)
    @property
    def variables(self):
        return [self.sum_sq, self.sum_w]
    
    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        return cfg
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MeanSquaredError":
        return cls(**config)
    
class MeanAbsoluteError(Metric):
    """Stateful Mean Absolute Error."""
    def __init__(self, name: str = 'mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_abs = self.add_variable('sum_abs', (), initializer=Zeros(), dtype=self.dtype)
        self.sum_w   = self.add_variable('sum_w',   (), initializer=Zeros(), dtype=self.dtype)

    def update_state(self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None):
        y_true = jnp.asarray(y_true, dtype=self.dtype)
        y_pred = jnp.asarray(y_pred, dtype=self.dtype)
        ae = jnp.abs(y_pred - y_true)
        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype).ravel()
            w = jnp.broadcast_to(w[:, None], ae.shape) if ae.ndim>1 else w
            sum_abs = jnp.sum(ae * w)
            sum_w   = jnp.sum(w)
        else:
            sum_abs = jnp.sum(ae)
            sum_w   = ae.size
        self.sum_abs.assign(self.sum_abs + sum_abs)
        self.sum_w .assign(self.sum_w  + sum_w)

    def result(self) -> jnp.ndarray:
        return self.sum_abs.value / (self.sum_w.value + 1e-8)

    @property
    def variables(self):
        return [self.sum_abs, self.sum_w]
    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        return cfg
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MeanAbsoluteError":
        return cls(**config)

class RootMeanSquaredError(Metric):
    """Stateful Root Mean Squared Error."""
    def __init__(self, name: str = 'rmse', **kwargs):
        super().__init__(name=name, **kwargs)
        self._mse = MeanSquaredError(name=name, **kwargs)

    def update_state(self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None):
        self._mse.update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        self._mse.reset_state()

    def result(self) -> jnp.ndarray:
        return jnp.sqrt(self._mse.result())

    @property
    def variables(self):
        return self._mse.variables

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RootMeanSquaredError":
        return cls(**config)


class R2Score(Metric):
    """Stateful Coefficient of Determination R^2."""
    def __init__(self, name: str = 'r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_w   = self.add_variable('sum_w',   (), initializer=Zeros(), dtype=self.dtype)
        self.sum_y   = self.add_variable('sum_y',   (), initializer=Zeros(), dtype=self.dtype)
        self.sum_yy  = self.add_variable('sum_yy',  (), initializer=Zeros(), dtype=self.dtype)
        self.sum_res = self.add_variable('sum_res', (), initializer=Zeros(), dtype=self.dtype)

    def update_state(self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None):
        y = jnp.asarray(y_true, dtype=self.dtype)
        p = jnp.asarray(y_pred, dtype=self.dtype)
        res = (y - p)**2
        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype).ravel()
            w = jnp.broadcast_to(w[:, None], y.shape) if y.ndim>1 else w
            sum_w   = jnp.sum(w)
            sum_y   = jnp.sum(y * w)
            sum_yy  = jnp.sum((y**2) * w)
            sum_res = jnp.sum(res * w)
        else:
            sum_w   = y.size
            sum_y   = jnp.sum(y)
            sum_yy  = jnp.sum(y**2)
            sum_res = jnp.sum(res)

        self.sum_w .assign(self.sum_w  + sum_w)
        self.sum_y .assign(self.sum_y  + sum_y)
        self.sum_yy.assign(self.sum_yy + sum_yy)
        self.sum_res.assign(self.sum_res + sum_res)

    def result(self) -> jnp.ndarray:
        sw = self.sum_w.value + 1e-8
        sy = self.sum_y.value
        syy= self.sum_yy.value
        ss_res = self.sum_res.value
        # SS_tot = sum_yy - sum_y^2 / sum_w
        ss_tot = syy - sy**2 / sw
        return 1 - ss_res / (ss_tot + 1e-8)
    
    @property
    def variables(self):
        return [self.sum_w, self.sum_y, self.sum_yy, self.sum_res]

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        return cfg
   

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "R2Score":
        return cls(**config)
