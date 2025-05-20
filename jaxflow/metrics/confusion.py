from typing import Optional, Any, Dict, Sequence

from jaxflow.core.variable import Variable
from jaxflow.initializers import Zeros
from jaxflow.metrics.metric import Metric

import jax.numpy as jnp
from typing import Optional




def average_fn(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    support: jnp.ndarray,
    average: Optional[str] = 'binary',
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Aggregate per-class metrics given per-class numerators and denominators.

    Args:
        numerator:    Array of shape (n_classes,) containing per-class numerators
                      (e.g. true positives for recall/precision).
        denominator:  Array of shape (n_classes,) containing per-class denominators
                      (e.g. actual positives for recall, predicted positives for precision).
        support:      Array of shape (n_classes,) containing the number of true
                      samples in each class (i.e. TP + FN).
        average:      One of {'binary', 'micro', 'macro', 'weighted', 'samples'} or
                      None. Default `'binary'`.
        epsilon:      Small constant to avoid division by zero.

    Returns:
        A scalar jnp.ndarray if `average` is not None, otherwise an array of
        shape (n_classes,) containing the per-class metric.
    """
    n_classes = numerator.shape[0]

    if average is None:
        return numerator / (denominator + epsilon)

    if average == 'binary':
        if n_classes != 2:
            raise ValueError("`binary` averaging is only valid for 2 classes.")
        # positive class assumed to be class index 1
        return numerator[1] / (denominator[1] + epsilon)

    if average == 'micro':
        # global numerator / global denominator
        return jnp.sum(numerator) / (jnp.sum(denominator) + epsilon)

    if average == 'macro':
        per_class = numerator / (denominator + epsilon)
        return jnp.mean(per_class)

    if average == 'weighted':
        per_class = numerator / (denominator + epsilon)
        weights = support / (jnp.sum(support) + epsilon)
        return jnp.sum(per_class * weights)

    if average == 'samples':
        # For multilabel metrics: would need per-sample values.
        raise NotImplementedError("`samples` average is not supported in this context.")

    raise ValueError(f"Unknown average type: {average}")



def true_positives(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    average: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> jnp.ndarray:
    # flatten & turn logits/one‐hots into class indices
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)
    if y_pred.ndim == y_true.ndim + 1:
        y_pred = jnp.argmax(y_pred, axis=-1)
    elif y_true.ndim == y_pred.ndim + 1:
        y_true = jnp.argmax(y_true, axis=-1)
    y_true = jnp.ravel(y_true)
    y_pred = jnp.ravel(y_pred)

    # infer num_classes
    if num_classes is None:
        num_classes = int(jnp.max(jnp.concatenate([y_true, y_pred])) + 1)

    one_hot_true = jnp.eye(num_classes, dtype=jnp.bool_)[y_true]
    one_hot_pred = jnp.eye(num_classes, dtype=jnp.bool_)[y_pred]
    mask = one_hot_true & one_hot_pred  # shape (n_samples, n_classes)

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, jnp.float32).ravel()
        w = jnp.broadcast_to(w[:, None], mask.shape)
        tp_counts = jnp.sum(mask.astype(jnp.float32) * w, axis=0)
        support = jnp.sum(one_hot_true.astype(jnp.float32) * w, axis=0)
    else:
        tp_counts = jnp.sum(mask, axis=0).astype(jnp.float32)
        support  = jnp.sum(one_hot_true, axis=0).astype(jnp.float32)

    if average is None:
        return tp_counts
    else:
        return average_fn(tp_counts, support, support, average)
class TruePositives(Metric):
    """
    Metric class for true positives, with support for sample_weight and averaging.
    """
    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = 'binary',
        name: str = 'true_positives',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.tp = self.add_variable('tp', (num_classes,), initializer=Zeros(), dtype=self.dtype)
        self.support = self.add_variable('support', (num_classes,), initializer=Zeros(), dtype=self.dtype)

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        # compute per-class TP counts (raw, no averaging)
        tp_inc = true_positives(
            y_true, y_pred,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )

        # compute support = actual positives per class
        y_true_arr = jnp.asarray(y_true)
        y_pred_arr = jnp.asarray(y_pred)
        # collapse one-hot or logits if needed
        if y_pred_arr.ndim == y_true_arr.ndim + 1:
            y_pred_arr = jnp.argmax(y_pred_arr, axis=-1)
        elif y_true_arr.ndim == y_pred_arr.ndim + 1:
            y_true_arr = jnp.argmax(y_true_arr, axis=-1)
        y_true_flat = jnp.ravel(y_true_arr)
        one_hot_true = jnp.eye(self.num_classes, dtype=jnp.bool_)[y_true_flat]

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype).ravel()
            w = jnp.broadcast_to(w[:, None], one_hot_true.shape)
            sup_inc = jnp.sum(one_hot_true.astype(self.dtype) * w, axis=0)
        else:
            sup_inc = jnp.sum(one_hot_true, axis=0).astype(self.dtype)

        # accumulate state
        self.tp.assign(self.tp + tp_inc)
        self.support.assign(self.support + sup_inc)

    def result(self) -> jnp.ndarray:
        tp_vals = self.tp.value
        sup_vals = self.support.value
        return average_fn(tp_vals, sup_vals, sup_vals, self.average)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            'num_classes': self.num_classes,
            'average': self.average,
        })
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TruePositives":
        return cls(**config)




def false_positives(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    average: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> jnp.ndarray:
    # same shape‐fixing as above...
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)
    if y_pred.ndim == y_true.ndim + 1:
        y_pred = jnp.argmax(y_pred, axis=-1)
    elif y_true.ndim == y_pred.ndim + 1:
        y_true = jnp.argmax(y_true, axis=-1)
    y_true = jnp.ravel(y_true)
    y_pred = jnp.ravel(y_pred)

    if num_classes is None:
        num_classes = int(jnp.max(jnp.concatenate([y_true, y_pred])) + 1)

    one_hot_true = jnp.eye(num_classes, dtype=jnp.bool_)[y_true]
    one_hot_pred = jnp.eye(num_classes, dtype=jnp.bool_)[y_pred]
    mask_fp = (~one_hot_true) & one_hot_pred

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, jnp.float32).ravel()
        w = jnp.broadcast_to(w[:, None], mask_fp.shape)
        fp_counts   = jnp.sum(mask_fp.astype(jnp.float32) * w, axis=0)
        pred_counts = jnp.sum(one_hot_pred.astype(jnp.float32) * w, axis=0)
    else:
        fp_counts   = jnp.sum(mask_fp, axis=0).astype(jnp.float32)
        pred_counts = jnp.sum(one_hot_pred, axis=0).astype(jnp.float32)

    if average is None:
        return fp_counts
    else:
        return average_fn(fp_counts, pred_counts, pred_counts, average)


class FalsePositives(Metric):
    """
    Metric class for false positives, with support for sample_weight and averaging.
    """
    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = 'binary',
        name: str = 'false_positives',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.fp = self.add_variable('fp', (num_classes,), initializer=Zeros(), dtype=self.dtype)
        self.pred_support = self.add_variable('pred_support', (num_classes,), initializer=Zeros(), dtype=self.dtype)

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        # compute per-class FP counts (raw, no averaging)
        fp_inc = false_positives(
            y_true, y_pred,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )

        # compute predicted support = predicted positives per class
        y_true_arr = jnp.asarray(y_true)
        y_pred_arr = jnp.asarray(y_pred)
        if y_pred_arr.ndim == y_true_arr.ndim + 1:
            y_pred_arr = jnp.argmax(y_pred_arr, axis=-1)
        elif y_true_arr.ndim == y_pred_arr.ndim + 1:
            y_true_arr = jnp.argmax(y_true_arr, axis=-1)
        y_pred_flat = jnp.ravel(y_pred_arr)
        one_hot_pred = jnp.eye(self.num_classes, dtype=jnp.bool_)[y_pred_flat]

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype).ravel()
            w = jnp.broadcast_to(w[:, None], one_hot_pred.shape)
            sup_inc = jnp.sum(one_hot_pred.astype(self.dtype) * w, axis=0)
        else:
            sup_inc = jnp.sum(one_hot_pred, axis=0).astype(self.dtype)

        # accumulate state
        self.fp.assign(self.fp + fp_inc)
        self.pred_support.assign(self.pred_support + sup_inc)

    def result(self) -> jnp.ndarray:
        fp_vals = self.fp.value
        sup_vals = self.pred_support.value
        return average_fn(fp_vals, sup_vals, sup_vals, self.average)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            'num_classes': self.num_classes,
            'average': self.average,
        })
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FalsePositives":
        return cls(**config)
    



def true_negatives(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    average: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> jnp.ndarray:
    """
    Computes per‐class true negatives and applies averaging.

    Returns raw counts if average=None, else aggregated value.
    """
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)

    # Collapse one-hot or logits if needed
    if y_pred.ndim == y_true.ndim + 1:
        y_pred = jnp.argmax(y_pred, axis=-1)
    elif y_true.ndim == y_pred.ndim + 1:
        y_true = jnp.argmax(y_true, axis=-1)
    elif y_pred.ndim != y_true.ndim:
        raise ValueError(f"y_true ndim ({y_true.ndim}) and y_pred ndim ({y_pred.ndim}) "
                         "must match or differ by 1.")

    y_true_flat = jnp.ravel(y_true)
    y_pred_flat = jnp.ravel(y_pred)

    # Infer number of classes
    if num_classes is None:
        num_classes = int(jnp.max(jnp.concatenate([y_true_flat, y_pred_flat])) + 1)

    # One-hot encode
    one_hot_true = jnp.eye(num_classes, dtype=jnp.bool_)[y_true_flat]
    one_hot_pred = jnp.eye(num_classes, dtype=jnp.bool_)[y_pred_flat]

    # Mask for true negatives
    tn_mask = (~one_hot_true) & (~one_hot_pred)
    # Negative support = cases where true label is negative
    neg_support_mask = ~one_hot_true

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32).ravel()
        w = jnp.broadcast_to(w[:, None], tn_mask.shape)
        tn_counts  = jnp.sum(tn_mask.astype(jnp.float32) * w, axis=0)
        sup_counts = jnp.sum(neg_support_mask.astype(jnp.float32) * w, axis=0)
    else:
        tn_counts  = jnp.sum(tn_mask, axis=0).astype(jnp.float32)
        sup_counts = jnp.sum(neg_support_mask, axis=0).astype(jnp.float32)

    if average is None:
        return tn_counts
    return average_fn(tn_counts, sup_counts, sup_counts, average)


class TrueNegatives(Metric):
    """
    Metric class for true negatives, with support for sample_weight and averaging.
    """
    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = 'binary',
        name: str = 'true_negatives',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.tn = self.add_variable('tn', (num_classes,), initializer=Zeros(), dtype=self.dtype)
        self.neg_support = self.add_variable('neg_support', (num_classes,), initializer=Zeros(), dtype=self.dtype)

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        tn_inc = true_negatives(
            y_true, y_pred,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )
        # negative support per class
        y_true_arr = jnp.asarray(y_true)
        if y_true_arr.ndim == jnp.asarray(y_pred).ndim + 1:
            y_true_arr = jnp.argmax(y_true_arr, axis=-1)
        neg_flat = jnp.ravel(y_true_arr)
        one_hot_true = jnp.eye(self.num_classes, dtype=jnp.bool_)[neg_flat]
        neg_mask = ~one_hot_true

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype).ravel()
            w = jnp.broadcast_to(w[:, None], neg_mask.shape)
            sup_inc = jnp.sum(neg_mask.astype(self.dtype) * w, axis=0)
        else:
            sup_inc = jnp.sum(neg_mask, axis=0).astype(self.dtype)

        self.tn.assign(self.tn + tn_inc)
        self.neg_support.assign(self.neg_support + sup_inc)

    def result(self) -> jnp.ndarray:
        tn_vals  = self.tn.value
        sup_vals = self.neg_support.value
        return average_fn(tn_vals, sup_vals, sup_vals, self.average)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({'num_classes': self.num_classes, 'average': self.average})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrueNegatives":
        return cls(**config)


def false_negatives(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    average: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> jnp.ndarray:
    """
    Computes per‐class false negatives and applies averaging.

    Returns raw counts if average=None, else aggregated value.
    """
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)

    # Collapse one-hot or logits if needed
    if y_pred.ndim == y_true.ndim + 1:
        y_pred = jnp.argmax(y_pred, axis=-1)
    elif y_true.ndim == y_pred.ndim + 1:
        y_true = jnp.argmax(y_true, axis=-1)
    elif y_pred.ndim != y_true.ndim:
        raise ValueError(f"y_true ndim ({y_true.ndim}) and y_pred ndim ({y_pred.ndim}) "
                         "must match or differ by 1.")

    y_true_flat = jnp.ravel(y_true)
    y_pred_flat = jnp.ravel(y_pred)

    # Infer number of classes
    if num_classes is None:
        num_classes = int(jnp.max(jnp.concatenate([y_true_flat, y_pred_flat])) + 1)

    one_hot_true = jnp.eye(num_classes, dtype=jnp.bool_)[y_true_flat]
    one_hot_pred = jnp.eye(num_classes, dtype=jnp.bool_)[y_pred_flat]

    fn_mask = one_hot_true & (~one_hot_pred)
    support_mask = one_hot_true

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, dtype=jnp.float32).ravel()
        w = jnp.broadcast_to(w[:, None], fn_mask.shape)
        fn_counts  = jnp.sum(fn_mask.astype(jnp.float32) * w, axis=0)
        sup_counts = jnp.sum(support_mask.astype(jnp.float32) * w, axis=0)
    else:
        fn_counts  = jnp.sum(fn_mask, axis=0).astype(jnp.float32)
        sup_counts = jnp.sum(support_mask, axis=0).astype(jnp.float32)

    if average is None:
        return fn_counts
    return average_fn(fn_counts, sup_counts, sup_counts, average)


class FalseNegatives(Metric):
    """
    Metric class for false negatives, with support for sample_weight and averaging.
    """
    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = 'binary',
        name: str = 'false_negatives',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.fn = self.add_variable('fn', (num_classes,), initializer=Zeros(), dtype=self.dtype)
        self.support = self.add_variable('support', (num_classes,), initializer=Zeros(), dtype=self.dtype)

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        fn_inc = false_negatives(
            y_true, y_pred,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )

        y_true_arr = jnp.asarray(y_true)
        if y_true_arr.ndim == jnp.asarray(y_pred).ndim + 1:
            y_true_arr = jnp.argmax(y_true_arr, axis=-1)
        sup_flat = jnp.ravel(y_true_arr)
        one_hot_true = jnp.eye(self.num_classes, dtype=jnp.bool_)[sup_flat]

        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype).ravel()
            w = jnp.broadcast_to(w[:, None], one_hot_true.shape)
            sup_inc = jnp.sum(one_hot_true.astype(self.dtype) * w, axis=0)
        else:
            sup_inc = jnp.sum(one_hot_true, axis=0).astype(self.dtype)

        self.fn.assign(self.fn + fn_inc)
        self.support.assign(self.support + sup_inc)

    def result(self) -> jnp.ndarray:
        fn_vals  = self.fn.value
        sup_vals = self.support.value
        return average_fn(fn_vals, sup_vals, sup_vals, self.average)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({'num_classes': self.num_classes, 'average': self.average})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FalseNegatives":
        return cls(**config)
    





def precision(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    average: Optional[str] = 'binary',
    num_classes: Optional[int] = None,
) -> jnp.ndarray:
    # 1) get raw counts
    tp = true_positives(y_true, y_pred,
                        sample_weight=sample_weight,
                        average=None,
                        num_classes=num_classes)
    fp = false_positives(y_true, y_pred,
                         sample_weight=sample_weight,
                         average=None,
                         num_classes=num_classes)
    denom = tp + fp

    # 2) true‐support for weighted averaging
    #    (number of actual positives per class)
    y_true_arr = jnp.asarray(y_true)
    if y_true_arr.ndim == jnp.asarray(y_pred).ndim + 1:
        y_true_arr = jnp.argmax(y_true_arr, axis=-1)
    y_true_flat = jnp.ravel(y_true_arr)
    if num_classes is None:
        num_classes = int(jnp.max(y_true_flat) + 1)
    one_hot_true = jnp.eye(num_classes, dtype=jnp.float32)[y_true_flat]

    if sample_weight is not None:
        w = jnp.asarray(sample_weight, jnp.float32).ravel()
        w = jnp.broadcast_to(w[:, None], one_hot_true.shape)
        support = jnp.sum(one_hot_true * w, axis=0)
    else:
        support = jnp.sum(one_hot_true, axis=0)

    # 3) aggregate
    return average_fn(tp, denom, support, average)


class Precision(Metric):
    """
    Precision metric: TP / (TP + FP) with support for sample_weight and averaging.
    """
    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = 'binary',
        name: str = 'precision',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.tp = self.add_variable('tp', (num_classes,), initializer=Zeros(), dtype=self.dtype)
        self.fp = self.add_variable('fp', (num_classes,), initializer=Zeros(), dtype=self.dtype)
        self.support = self.add_variable('support', (num_classes,), initializer=Zeros(), dtype=self.dtype)

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        y_true_arr = jnp.asarray(y_true)
        y_pred_arr = jnp.asarray(y_pred)

        # same shape logic
        if y_pred_arr.ndim == y_true_arr.ndim + 1:
            y_pred_arr = jnp.argmax(y_pred_arr, axis=-1)
        elif y_true_arr.ndim == y_pred_arr.ndim + 1:
            y_true_arr = jnp.argmax(y_true_arr, axis=-1)
        elif y_pred_arr.ndim != y_true_arr.ndim:
            raise ValueError(
                f"y_true ndim ({y_true_arr.ndim}) and y_pred ndim ({y_pred_arr.ndim}) "
                "must either match or differ by exactly one."
            )

        # get batch‐level TP/FP counts
        tp_inc = true_positives(
            y_true_arr, y_pred_arr,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )
        fp_inc = false_positives(
            y_true_arr, y_pred_arr,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )

        # recompute per‐class support
        y_true_flat = jnp.ravel(y_true_arr)
        one_hot_true = jnp.eye(self.num_classes, dtype=jnp.bool_)[y_true_flat]
        if sample_weight is not None:
            w = jnp.asarray(sample_weight, dtype=self.dtype).ravel()
            w = jnp.broadcast_to(w[:, None], one_hot_true.shape)
            sup_inc = jnp.sum(one_hot_true.astype(self.dtype) * w, axis=0)
        else:
            sup_inc = jnp.sum(one_hot_true, axis=0).astype(self.dtype)

        # accumulate
        self.tp.assign(self.tp + tp_inc)
        self.fp.assign(self.fp + fp_inc)
        self.support.assign(self.support + sup_inc)

    def result(self) -> jnp.ndarray:
        tp_vals = self.tp.value
        fp_vals = self.fp.value
        denom = tp_vals + fp_vals
        sup_vals = self.support.value
        return average_fn(tp_vals, denom, sup_vals, self.average)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({'num_classes': self.num_classes, 'average': self.average})
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Precision":
        return cls(**config)
    





def recall(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    average: Optional[str] = 'binary',
    num_classes: Optional[int] = None,
) -> jnp.ndarray:
    """
    Computes recall = TP / (TP + FN), with optional averaging:
      - TP = true_positives(..., average=None)
      - FN = false_negatives(..., average=None)
      - actual positives per class = TP + FN
    """
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)

    # collapse logits/one-hot to class indices if needed
    if y_pred.ndim == y_true.ndim + 1:
        y_pred = jnp.argmax(y_pred, axis=-1)
    elif y_true.ndim == y_pred.ndim + 1:
        y_true = jnp.argmax(y_true, axis=-1)
    elif y_pred.ndim != y_true.ndim:
        raise ValueError(
            f"y_true ndim ({y_true.ndim}) and y_pred ndim ({y_pred.ndim}) "
            "must match or differ by exactly one."
        )

    # raw per-class counts
    tp = true_positives(
        y_true, y_pred,
        sample_weight=sample_weight,
        average=None,
        num_classes=num_classes
    )
    fn = false_negatives(
        y_true, y_pred,
        sample_weight=sample_weight,
        average=None,
        num_classes=num_classes
    )
    denom = tp + fn

    # for weighted average support = actual positives = denom
    support = denom

    return average_fn(tp, denom, support, average)

class Recall(Metric):
    """
    Recall metric: TP / (TP + FN), supporting sample_weight and averaging modes.
    """
    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = 'binary',
        name: str = 'recall',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        # state variables: per-class TP and actual positives (support)
        self.tp = self.add_variable(
            'tp', (num_classes,), initializer=Zeros(), dtype=self.dtype
        )
        self.support = self.add_variable(
            'support', (num_classes,), initializer=Zeros(), dtype=self.dtype
        )

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        # raw TP and FN counts
        tp_inc = true_positives(
            y_true, y_pred,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )
        fn_inc = false_negatives(
            y_true, y_pred,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )

        # actual positives per class = TP + FN
        sup_inc = tp_inc + fn_inc

        # accumulate counts
        self.tp.assign(self.tp + tp_inc)
        self.support.assign(self.support + sup_inc)

    def result(self) -> jnp.ndarray:
        tp_vals = self.tp.value
        sup_vals = self.support.value
        return average_fn(tp_vals, sup_vals, sup_vals, self.average)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            'num_classes': self.num_classes,
            'average': self.average,
        })
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Recall":
        return cls(**config)
    


def f1_score(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    average: Optional[str] = 'binary',
    num_classes: Optional[int] = None,
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    F1 = 2 * (precision * recall) / (precision + recall + epsilon)
    Uses our precision() and recall() helpers.
    """
    p = precision(
        y_true, y_pred,
        sample_weight=sample_weight,
        average=average,
        num_classes=num_classes
    )
    r = recall(
        y_true, y_pred,
        sample_weight=sample_weight,
        average=average,
        num_classes=num_classes
    )
    return 2 * p * r / (p + r + epsilon)


class F1Score(Metric):
    """
    F1 metric: 2 * precision * recall / (precision + recall).
    Internally delegates to the precision and recall Metric classes.
    """
    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = 'binary',
        name: str = 'f1_score',
        epsilon: float = 1e-8,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.epsilon = epsilon
        # sub‐metrics to accumulate TP/FP/FN
        from jaxflow.metrics import Precision, Recall  # adjust import path
        self._precision = Precision(num_classes, average, name=f"{name}_prec", **kwargs)
        self._recall    = Recall(num_classes, average, name=f"{name}_rec",  **kwargs)

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        # update both sub‐metrics
        self._precision.update_state(y_true, y_pred, sample_weight)
        self._recall.update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        self._precision.reset_state()
        self._recall.reset_state()

    @property
    def variables(self):
        # expose both precision and recall variables
        return self._precision.variables + self._recall.variables

    def result(self) -> jnp.ndarray:
        p = self._precision.result()
        r = self._recall.result()
        return 2 * p * r / (p + r + self.epsilon)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_classes": self.num_classes,
            "average": self.average,
            "epsilon": self.epsilon,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    




def specificity(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    average: Optional[str] = 'binary',
    num_classes: Optional[int] = None,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """
    Computes specificity = TN / (TN + FP), with optional averaging:
      - TN = true_negatives(..., average=None)
      - FP = false_positives(..., average=None)
      - actual negatives per class = TN + FP
    """
    # raw per-class counts
    tn = true_negatives(
        y_true, y_pred,
        sample_weight=sample_weight,
        average=None,
        num_classes=num_classes
    )
    fp = false_positives(
        y_true, y_pred,
        sample_weight=sample_weight,
        average=None,
        num_classes=num_classes
    )
    denom = tn + fp + epsilon
    support = tn + fp

    return average_fn(tn, denom, support, average)


class Specificity(Metric):
    """
    Specificity metric: TN / (TN + FP), supporting sample_weight and averaging modes.
    """
    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = 'binary',
        name: str = 'specificity',
        epsilon: float = 1e-8,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average     = average
        self.epsilon     = epsilon
        # state: per-class true negatives and false positives
        self.tn = self.add_variable(
            'tn', (num_classes,), initializer=Zeros(), dtype=self.dtype
        )
        self.fp = self.add_variable(
            'fp', (num_classes,), initializer=Zeros(), dtype=self.dtype
        )

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        tn_inc = true_negatives(
            y_true, y_pred,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )
        fp_inc = false_positives(
            y_true, y_pred,
            sample_weight=sample_weight,
            average=None,
            num_classes=self.num_classes
        )
        self.tn.assign(self.tn + tn_inc)
        self.fp.assign(self.fp + fp_inc)

    def result(self) -> jnp.ndarray:
        tn_vals = self.tn.value
        fp_vals = self.fp.value
        denom   = tn_vals + fp_vals + self.epsilon
        support = tn_vals + fp_vals
        return average_fn(tn_vals, denom, support, self.average)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            'num_classes': self.num_classes,
            'average': self.average,
            'epsilon': self.epsilon,
        })
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Specificity":
        return cls(**config)
    

def sensitivity(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: Optional[jnp.ndarray] = None,
    average: Optional[str] = 'binary',
    num_classes: Optional[int] = None
) -> jnp.ndarray:
    """
    Sensitivity (a.k.a. recall or true positive rate):
      - sensitivity = TP / (TP + FN)
      - delegates to the recall() helper
    """
    return recall(
        y_true, y_pred,
        sample_weight=sample_weight,
        average=average,
        num_classes=num_classes
    )


class Sensitivity(Recall):
    """
    Sensitivity metric class (alias of Recall), supporting sample_weight and averaging.
    """
    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = 'binary',
        name: str = 'sensitivity',
        **kwargs
    ):
        # Just call Recall.__init__ with a different default name
        super().__init__(num_classes=num_classes, average=average, name=name, **kwargs)
    
    # Inherits update_state, result, get_config, from_config from Recall









def confusion_matrix(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    labels: Optional[Sequence[Any]] = None,
    sample_weight: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Compute confusion matrix like sklearn.metrics.confusion_matrix,
    but using JAX-friendly, functional updates.
    """
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)
    # Collapse one-hot/logits if needed
    if y_pred.ndim == y_true.ndim + 1:
        y_pred = jnp.argmax(y_pred, axis=-1)
    elif y_true.ndim == y_pred.ndim + 1:
        y_true = jnp.argmax(y_true, axis=-1)
    elif y_true.ndim != y_pred.ndim:
        raise ValueError(f"y_true ndim ({y_true.ndim}) and y_pred ndim ({y_pred.ndim}) "
                         "must match or differ by exactly one.")
    yt = jnp.ravel(y_true)
    yp = jnp.ravel(y_pred)

    # Determine label set
    if labels is None:
        labels = jnp.unique(jnp.concatenate([yt, yp]))
    else:
        labels = jnp.asarray(labels)
    n_labels = labels.shape[0]

    # Map samples to integer indices in [0, n_labels)
    # Build boolean (n_samples, n_labels) masks, then argmax
    row_idx = jnp.argmax(yt[:, None] == labels[None, :], axis=1)
    col_idx = jnp.argmax(yp[:, None] == labels[None, :], axis=1)

    # Prepare weights
    if sample_weight is not None:
        w = jnp.asarray(sample_weight).ravel()
    else:
        w = jnp.ones_like(row_idx, dtype=jnp.int32)

    # Functional accumulation
    cm = jnp.zeros((n_labels, n_labels), dtype=jnp.int32)
    cm = cm.at[row_idx, col_idx].add(w)
    return cm


class ConfusionMatrix(Metric):
    """
    Stateful confusion matrix metric: accumulates counts over batches.
    """
    def __init__(
        self,
        num_classes: int,
        name: str = 'confusion_matrix',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # state: a num_classes x num_classes matrix
        self.matrix = self.add_variable(
            'matrix',
            (num_classes, num_classes),
            initializer=Zeros(),
            dtype=jnp.int32
        )

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ):
        # Compute batch confusion matrix
        cm_batch = confusion_matrix(
            jnp.asarray(y_true),
            jnp.asarray(y_pred),
            labels=jnp.arange(self.num_classes),
            sample_weight=sample_weight
        )
        # Accumulate: functional update on the variable
        new_matrix = self.matrix.value + cm_batch
        self.matrix.assign(new_matrix)

    def result(self) -> jnp.ndarray:
        return self.matrix.value

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({'num_classes': self.num_classes})
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> "ConfusionMatrix":
        return cls(**config)
