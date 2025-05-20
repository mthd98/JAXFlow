from typing import Any, List, Optional, Dict
import jax.numpy as jnp

from jaxflow.core.variable import Variable
from jaxflow.initializers import Zeros
from .metric import Metric  # adjust import path as needed

def accuracy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Computes (optionally weighted) accuracy:
      sum(matches * weight) / sum(weight)
    If sample_weight is None, weight=1 for every element.
    """
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred, dtype=y_true.dtype)
    # flatten
    y_true = jnp.ravel(y_true)
    y_pred = jnp.ravel(y_pred)
    match = jnp.equal(y_true, y_pred).astype(jnp.float32)

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32)
        w = jnp.broadcast_to(w, match.shape)
        return jnp.sum(match * w) / (jnp.sum(w) + 1e-8)
    else:
        return jnp.mean(match)


class Accuracy(Metric):
    """
    Accuracy metric: accumulates correct predictions and total weights.
    """
    def __init__(self, name: str = 'accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        # tracks numerator and denominator
        self.total_correct = self.add_variable(
            'total_correct', (), initializer=Zeros(), dtype=self.dtype
        )
        self.total_weight = self.add_variable(
            'total_weight', (), initializer=Zeros(), dtype=self.dtype
        )

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        # compute weighted correct count and weight sum
        y_true_arr = jnp.asarray(y_true)
        y_pred_arr = jnp.asarray(y_pred, dtype=y_true_arr.dtype)
        y_true_flat = jnp.ravel(y_true_arr)
        y_pred_flat = jnp.ravel(y_pred_arr)
        matches = jnp.equal(y_true_flat, y_pred_flat).astype(self.dtype)

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype)
            w = jnp.broadcast_to(w, matches.shape)
            correct = jnp.sum(matches * w)
            weight_sum = jnp.sum(w)
        else:
            correct = jnp.sum(matches)
            weight_sum = matches.size

        # update state
        self.total_correct.assign(self.total_correct + correct)
        self.total_weight.assign(self.total_weight + weight_sum)

    def result(self) -> jnp.ndarray:
        return self.total_correct / (self.total_weight + 1e-8)






def binary_accuracy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    threshold: float = 0.5,
    sample_weight: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Computes binary accuracy:
      - Binarize predictions at `threshold`
      - accuracy = sum(equal(y_true, y_pred_bin) * weight) / sum(weight)
    """
    y_true = jnp.asarray(y_true).astype(jnp.bool_)
    y_pred = jnp.asarray(y_pred)
    # binarize predictions
    y_pred_bin = jnp.greater(y_pred, threshold)
    matches = jnp.equal(y_true, y_pred_bin).astype(jnp.float32)

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32)
        w = jnp.broadcast_to(w, matches.shape)
        return jnp.sum(matches * w) / (jnp.sum(w) + 1e-8)
    else:
        return jnp.mean(matches)


class BinaryAccuracy(Metric):
    """
    Binary accuracy metric: accumulates correct predictions and total weights.
    """
    def __init__(self, name: str = 'binary_accuracy', threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_correct = self.add_variable(
            'total_correct', (), initializer=Zeros(), dtype=self.dtype
        )
        self.total_weight = self.add_variable(
            'total_weight', (), initializer=Zeros(), dtype=self.dtype
        )

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        y_true_arr = jnp.asarray(y_true).astype(jnp.bool_)
        y_pred_arr = jnp.asarray(y_pred)
        y_pred_bin = jnp.greater(y_pred_arr, self.threshold)
        matches = jnp.equal(y_true_arr, y_pred_bin).astype(self.dtype)

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype)
            w = jnp.broadcast_to(w, matches.shape)
            correct = jnp.sum(matches * w)
            weight_sum = jnp.sum(w)
        else:
            correct = jnp.sum(matches)
            weight_sum = matches.size

        self.total_correct.assign(self.total_correct + correct)
        self.total_weight.assign(self.total_weight + weight_sum)

    def result(self) -> jnp.ndarray:
        return self.total_correct / (self.total_weight + 1e-8)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({'threshold': self.threshold})
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> "BinaryAccuracy":
        return cls(**config)




def categorical_accuracy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Computes categorical accuracy:
      - Takes argmax over the last axis for predictions & labels
      - accuracy = sum(equal) / count  (or weighted)
    """
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)
    # class indices
    true_idx = jnp.argmax(y_true, axis=-1)
    pred_idx = jnp.argmax(y_pred, axis=-1)
    matches = jnp.equal(true_idx, pred_idx).astype(jnp.float32)

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32)
        w = jnp.broadcast_to(w, matches.shape)
        return jnp.sum(matches * w) / (jnp.sum(w) + 1e-8)
    else:
        return jnp.mean(matches)


class CategoricalAccuracy(Metric):
    """
    Categorical accuracy metric: accumulates correct predictions over total samples,
    supporting optional sample_weight.
    """
    def __init__(self, name: str = 'categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        # numerator and denominator
        self.total_correct = self.add_variable(
            'total_correct', (), initializer=Zeros(), dtype=self.dtype
        )
        self.total_weight = self.add_variable(
            'total_weight', (), initializer=Zeros(), dtype=self.dtype
        )

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        y_true_arr = jnp.asarray(y_true)
        y_pred_arr = jnp.asarray(y_pred)
        true_idx = jnp.argmax(y_true_arr, axis=-1)
        pred_idx = jnp.argmax(y_pred_arr, axis=-1)
        matches = jnp.equal(true_idx, pred_idx).astype(self.dtype)

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype)
            w = jnp.broadcast_to(w, matches.shape)
            correct = jnp.sum(matches * w)
            weight_sum = jnp.sum(w)
        else:
            correct = jnp.sum(matches)
            weight_sum = matches.size

        self.total_correct.assign(self.total_correct + correct)
        self.total_weight.assign(self.total_weight + weight_sum)

    def result(self) -> jnp.ndarray:
        return self.total_correct / (self.total_weight + 1e-8)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        return cfg  # no extra args

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CategoricalAccuracy":
        return cls(**config)
    




def sparse_categorical_accuracy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Computes sparse categorical accuracy:
      - y_true: integer class indices, shape (...) 
      - y_pred: logits or probabilities, shape (..., num_classes)
      - accuracy = sum(matches * weight) / sum(weight)
    """
    # ensure arrays
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)
    # predicted class = argmax over last axis
    pred_idx = jnp.argmax(y_pred, axis=-1)
    # compare to true indices
    matches = jnp.equal(y_true, pred_idx).astype(jnp.float32)

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32)
        w = jnp.broadcast_to(w, matches.shape)
        return jnp.sum(matches * w) / (jnp.sum(w) + 1e-8)
    else:
        return jnp.mean(matches)


class SparseCategoricalAccuracy(Metric):
    """
    Sparse categorical accuracy metric: accumulates correct predictions and total weights.
    """
    def __init__(self, name: str = 'sparse_categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        # numerator: weighted count of correct predictions
        self.total_correct = self.add_variable(
            'total_correct', (), initializer=Zeros(), dtype=self.dtype
        )
        # denominator: sum of weights or sample counts
        self.total_weight = self.add_variable(
            'total_weight', (), initializer=Zeros(), dtype=self.dtype
        )

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        # convert inputs
        y_true_arr = jnp.asarray(y_true)
        y_pred_arr = jnp.asarray(y_pred)
        # compute matches
        pred_idx = jnp.argmax(y_pred_arr, axis=-1)
        matches = jnp.equal(y_true_arr, pred_idx).astype(self.dtype)

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype)
            w = jnp.broadcast_to(w, matches.shape)
            correct = jnp.sum(matches * w)
            weight_sum = jnp.sum(w)
        else:
            correct = jnp.sum(matches)
            weight_sum = matches.size

        # update state
        self.total_correct.assign(self.total_correct + correct)
        self.total_weight.assign(self.total_weight + weight_sum)

    def result(self) -> jnp.ndarray:
        return self.total_correct / (self.total_weight + 1e-8)

    def get_config(self) -> Dict[str, Any]:
        # no extra args beyond name/dtype
        return super().get_config()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparseCategoricalAccuracy":
        return cls(**config)
    


def top_k_categorical_accuracy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    k: int = 5,
    sample_weight: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Computes top-k categorical accuracy:
      - y_true: one-hot labels or integer labels
      - y_pred: logits or probabilities, shape (..., num_classes)
      - accuracy = sum(weighted matches) / sum(weights)
    """
    y_pred = jnp.asarray(y_pred)
    # If y_true is one-hot, convert to indices
    y_true = jnp.asarray(y_true)
    if y_true.ndim == y_pred.ndim:
        # assume one-hot on last axis
        true_idx = jnp.argmax(y_true, axis=-1)
    else:
        true_idx = y_true.astype(jnp.int32)

    # Get indices of top-k predictions
    # argsort ascending, take last k entries
    topk = jnp.argsort(y_pred, axis=-1)[..., -k:]
    # For each sample, check if true_idx is in topk
    # Expand dims of true_idx for comparison
    true_exp = jnp.expand_dims(true_idx, axis=-1)
    matches = jnp.any(topk == true_exp, axis=-1).astype(jnp.float32)

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32)
        w = jnp.broadcast_to(w, matches.shape)
        return jnp.sum(matches * w) / (jnp.sum(w) + 1e-8)
    else:
        return jnp.mean(matches)


class TopKCategoricalAccuracy(Metric):
    """
    Top-k categorical accuracy metric: accumulates correct top-k predictions
    and total weights.
    """
    def __init__(
        self,
        name: str = 'top_k_categorical_accuracy',
        k: int = 5,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total_correct = self.add_variable(
            'total_correct', (), initializer=Zeros(), dtype=self.dtype
        )
        self.total_weight = self.add_variable(
            'total_weight', (), initializer=Zeros(), dtype=self.dtype
        )

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        # Convert inputs
        y_pred_arr = jnp.asarray(y_pred)
        y_true_arr = jnp.asarray(y_true)
        if y_true_arr.ndim == y_pred_arr.ndim:
            true_idx = jnp.argmax(y_true_arr, axis=-1)
        else:
            true_idx = y_true_arr.astype(jnp.int32)

        topk = jnp.argsort(y_pred_arr, axis=-1)[..., -self.k:]
        true_exp = jnp.expand_dims(true_idx, axis=-1)
        matches = jnp.any(topk == true_exp, axis=-1).astype(self.dtype)

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype)
            w = jnp.broadcast_to(w, matches.shape)
            correct = jnp.sum(matches * w)
            weight_sum = jnp.sum(w)
        else:
            correct = jnp.sum(matches)
            weight_sum = matches.size

        self.total_correct.assign(self.total_correct + correct)
        self.total_weight.assign(self.total_weight + weight_sum)

    def result(self) -> jnp.ndarray:
        return self.total_correct / (self.total_weight + 1e-8)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({'k': self.k})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TopKCategoricalAccuracy":
        return cls(**config)
    


def sparse_top_k_categorical_accuracy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    k: int = 5,
    sample_weight: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Computes sparse top-k categorical accuracy:
      - y_true: integer class indices, shape (...)
      - y_pred: logits or probabilities, shape (..., num_classes)
      - accuracy = sum(matches * weight) / sum(weight)
    """
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)
    # Get top-k predicted class indices
    topk = jnp.argsort(y_pred, axis=-1)[..., -k:]
    # Expand true labels to compare against topk
    true_idx = y_true.astype(jnp.int32)[..., None]
    matches = jnp.any(topk == true_idx, axis=-1).astype(jnp.float32)

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32)
        w = jnp.broadcast_to(w, matches.shape)
        return jnp.sum(matches * w) / (jnp.sum(w) + 1e-8)
    else:
        return jnp.mean(matches)


class SparseTopKCategoricalAccuracy(Metric):
    """
    Sparse top-k categorical accuracy metric: accumulates correct top-k predictions
    and total weights.
    """
    def __init__(
        self,
        name: str = 'sparse_top_k_categorical_accuracy',
        k: int = 5,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total_correct = self.add_variable(
            'total_correct', (), initializer=Zeros(), dtype=self.dtype
        )
        self.total_weight = self.add_variable(
            'total_weight', (), initializer=Zeros(), dtype=self.dtype
        )

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        y_true_arr = jnp.asarray(y_true)
        y_pred_arr = jnp.asarray(y_pred)
        topk = jnp.argsort(y_pred_arr, axis=-1)[..., -self.k:]
        true_idx = y_true_arr.astype(jnp.int32)[..., None]
        matches = jnp.any(topk == true_idx, axis=-1).astype(self.dtype)

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype)
            w = jnp.broadcast_to(w, matches.shape)
            correct = jnp.sum(matches * w)
            weight_sum = jnp.sum(w)
        else:
            correct = jnp.sum(matches)
            weight_sum = matches.size

        self.total_correct.assign(self.total_correct + correct)
        self.total_weight.assign(self.total_weight + weight_sum)

    def result(self) -> jnp.ndarray:
        return self.total_correct / (self.total_weight + 1e-8)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({'k': self.k})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparseTopKCategoricalAccuracy":
        return cls(**config)