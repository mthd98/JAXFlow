import jax
import jax.numpy as jnp
from jaxflow.core.auto_name import AutoNameMixin

# Metaclass to intercept instantiation keyword arguments for Loss.
class LossMeta(type):
    def __call__(cls, *args, **kwargs):
        # Extract parameters intended for the Loss base class.
        loss_params = {}
        for key in ["reduction", "name", "dtype"]:
            if key in kwargs:
                loss_params[key] = kwargs.pop(key)
        # Create the instance using the subclass's __init__.
        instance = super().__call__(*args, **kwargs)
        # Reinitialize the base Loss attributes using the extracted parameters.
        Loss.__init__(instance, **loss_params)
        return instance

class Loss(AutoNameMixin, metaclass=LossMeta):
    """
    Base class for loss functions in a JAX/Optax workflow.

    This abstract class provides a consistent API for implementing custom losses
    compatible with JAX, Optax, and JAXFlow training workflows. Modeled after
    Keras's `Loss` base class, it supports:
      - Configurable reduction mode, dtype, and name
      - Sample weighting and masking (with flexible shape handling)
      - Automatic reduction over batch and/or sample dimensions
      - Easy subclassing for custom elementwise loss logic

    Args:
        name (str, optional): Optional name for the loss instance. If None, a unique name is generated.
        reduction (str, optional): Reduction mode to apply to the loss. Supported options:
            "sum_over_batch_size" (default), "sum", "mean", "mean_with_sample_weight", "none", or None.
        dtype (jnp.dtype, optional): Data type for loss computations. Defaults to jnp.float32.

    Inputs:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Model predictions.
        sample_weight (array-like, optional): Optional per-sample weights for the loss.
        mask (array-like, optional): Optional boolean mask to apply to y_true/y_pred.

    Input shape:
        Arbitrary shapes as long as `call(y_true, y_pred)` produces a per-sample loss.
        Sample weights and mask should be broadcastable to the loss shape.

    Output shape:
        Scalar, unless reduction="none", in which case the unreduced loss array is returned.

    Attributes:
        name (str): Name for the loss instance.
        reduction (str): Reduction method ("sum_over_batch_size", "sum", "mean", etc.).
        dtype (jnp.dtype): Data type used for computations.

    Example:
        ```python
        import jax.numpy as jnp
        from jaxflow.losses.loss import Loss

        class MySquaredError(Loss):
            def call(self, y_true, y_pred):
                return (y_true - y_pred) ** 2

        loss_fn = MySquaredError(reduction="mean")
        y_true = jnp.array([1.0, 2.0, 3.0])
        y_pred = jnp.array([1.1, 2.2, 2.9])
        loss = loss_fn(y_true, y_pred)
        print(loss)
        ```

    Notes:
        - To implement a new loss, override the `call(self, y_true, y_pred)` method in a subclass.
        - Sample weights and mask are automatically broadcast and applied if provided.
        - The reduction mode controls how the final scalar loss is computed from per-sample values.
        - Compatible with JAX JIT/vmap/pmap for functional workflows.

    Raises:
        NotImplementedError: If `call()` is not implemented in a subclass.
        ValueError: For invalid reduction arguments or incompatible input shapes.

    See Also:
        - Keras Loss base class: https://keras.io/api/losses/
        - Optax loss functions: https://optax.readthedocs.io/en/latest/api.html
    """


    @staticmethod
    def _standardize_reduction(reduction):
        allowed = {
            "sum_over_batch_size",
            "sum",
            None,
            "none",
            "mean",
            "mean_with_sample_weight",
        }
        if reduction not in allowed:
            raise ValueError(
                f"Invalid value for argument `reduction`. Expected one of {allowed}. Received: {reduction}"
            )
        return reduction

    @staticmethod
    def _squeeze_or_expand_to_same_rank(x1, x2, expand_rank_1=True):
        x1_rank = len(x1.shape)
        x2_rank = len(x2.shape)
        if x1_rank == x2_rank:
            return x1, x2
        if x1_rank == x2_rank + 1:
            if x1.shape[-1] == 1:
                if x2_rank == 1 and expand_rank_1:
                    x2 = jnp.expand_dims(x2, axis=-1)
                else:
                    x1 = jnp.squeeze(x1, axis=-1)
        elif x2_rank == x1_rank + 1:
            if x2.shape[-1] == 1:
                if x1_rank == 1 and expand_rank_1:
                    x1 = jnp.expand_dims(x1, axis=-1)
                else:
                    x2 = jnp.squeeze(x2, axis=-1)
        return x1, x2

    @staticmethod
    def _scale_loss_for_distribution(value):
        # For JAX, we simply return the value (or apply custom scaling if needed).
        return value

    @staticmethod
    def _reduce_values(values, sample_weight=None, reduction="sum_over_batch_size", dtype=jnp.float32):
        if reduction is None or reduction == "none" or values.ndim == 0 or values.size == 0:
            return values
        loss = jnp.sum(values)
        if reduction in ("sum_over_batch_size", "mean", "mean_with_sample_weight"):
            if reduction == "mean_with_sample_weight" and sample_weight is not None:
                divisor = jnp.sum(sample_weight)
            else:
                divisor = values.size
            loss = jnp.where(divisor != 0, loss / divisor, 0.0)
            loss = Loss._scale_loss_for_distribution(loss)
        return loss

    @staticmethod
    def _apply_mask(sample_weight, mask, dtype=jnp.float32, reduction="sum_over_batch_size"):
        if mask is not None:
            mask = mask.astype(dtype)
            if reduction in ("mean", "sum_over_batch_size"):
                total = mask.size
                valid = jnp.sum(mask)
                mask = mask * (total / (valid + 1e-7))
            if sample_weight is not None:
                sample_weight, mask = Loss._squeeze_or_expand_to_same_rank(sample_weight, mask)
                sample_weight = sample_weight * mask
            else:
                sample_weight = mask
        return sample_weight

    @staticmethod
    def _reduce_weighted_values(values, sample_weight=None, mask=None, reduction="sum_over_batch_size", dtype=jnp.float32):
        values = jnp.asarray(values, dtype=dtype)
        if sample_weight is not None:
            sample_weight = jnp.asarray(sample_weight, dtype=dtype)
        if mask is not None:
            mask = jnp.asarray(mask, dtype=dtype)
        sample_weight = Loss._apply_mask(sample_weight, mask, dtype=dtype, reduction=reduction)
        if sample_weight is not None:
            values, sample_weight = Loss._squeeze_or_expand_to_same_rank(values, sample_weight)
            values = values * sample_weight
        return Loss._reduce_values(values, sample_weight, reduction, dtype=dtype)

    def __init__(self, name=None, reduction="sum_over_batch_size", dtype=jnp.float32):
        self.name = self.auto_name(name)
        self.reduction = Loss._standardize_reduction(reduction)
        self.dtype = dtype

    def __call__(self, y_true, y_pred, sample_weight=None, mask=None):
        # Convert inputs to JAX arrays with the proper dtype.
        y_pred = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=self.dtype), y_pred)
        y_true = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=self.dtype), y_true)
        # Compute the raw per-sample loss using the subclass’s implementation.
        losses = self.call(y_true, y_pred)
        # Apply sample weighting, masking, and reduction.
        return Loss._reduce_weighted_values(
            losses,
            sample_weight=sample_weight,
            mask=mask,
            reduction=self.reduction,
            dtype=self.dtype,
        )

    def call(self, y_true, y_pred):
        """
        Subclasses must implement this method to compute the raw per-sample loss.
        """
        raise NotImplementedError("Subclasses must implement the call method.")

    def get_config(self):
        return {"name": self.name, "reduction": self.reduction, "dtype": self.dtype}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _obj_type(self):
        return "Loss"
