import jax
import jax.numpy as jnp

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

class Loss(metaclass=LossMeta):
    """
    Base class for loss functions in a JAX/Optax workflow.

    This class is inspired by Keras’s Loss base class and supports:
      - Configurable `name`, `reduction`, and `dtype`
      - Converting inputs to the proper dtype
      - Optional sample weighting and masking, followed by reduction

    Args:
      name: Optional name for the loss instance.
      reduction: Reduction mode to apply. Supported options are:
          "sum_over_batch_size", "sum", "mean", "mean_with_sample_weight", "none", or None.
      dtype: The dtype for loss computations. Defaults to jnp.float32.
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
        self.name = name or self.__class__.__name__
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
