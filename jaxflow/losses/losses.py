
import jax
import jax.numpy as jnp
from jaxflow.losses.loss import Loss

class BinaryCrossentropy(Loss):
    """
    Computes the Binary Crossentropy loss in a shape-agnostic way.

    Args:
      from_logits: Boolean, whether y_pred are logits. If True, applies sigmoid.
      label_smoothing: Float in [0, 1]. If > 0, smooth the labels.
      **kwargs: Additional keyword arguments forwarded to the Loss base class.
    """
    def __init__(self, from_logits=False, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # 1) Label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 2) Logits → probabilities
        if self.from_logits:
            y_pred = jax.nn.sigmoid(y_pred)

        # 3) Make ndim equal by appending singleton dims to the end
        if y_true.ndim < y_pred.ndim:
            y_true = y_true.reshape(y_true.shape + (1,) * (y_pred.ndim - y_true.ndim))
        elif y_pred.ndim < y_true.ndim:
            y_pred = y_pred.reshape(y_pred.shape + (1,) * (y_true.ndim - y_pred.ndim))

        # 4) Now broadcast to identical shapes
        try:
            y_true, y_pred = jnp.broadcast_arrays(y_true, y_pred)
        except ValueError as e:
            raise ValueError(
                f"Cannot align shapes y_true{y_true.shape} vs y_pred{y_pred.shape}"
            ) from e

        # 5) Clip to avoid log(0)
        eps = 1e-7
        y_pred = jnp.clip(y_pred, eps, 1 - eps)

        # 6) Compute binary crossentropy
        loss = - (y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))
        return loss

# =================== BinaryFocalCrossentropy ===================


class BinaryFocalCrossentropy(Loss):
    """
    Computes the Binary Focal Crossentropy loss in a shape-agnostic way.

    Args:
      gamma: Focusing parameter for modulating factor (default 2.0).
      alpha: Balancing parameter (default 0.25).
      from_logits: Boolean, whether y_pred are logits.
      label_smoothing: Float in [0, 1]. If > 0, smooth the labels.
      **kwargs: Additional keyword arguments forwarded to the Loss base class.
    """
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # 1) Label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 2) Logits → probabilities
        if self.from_logits:
            y_pred = jax.nn.sigmoid(y_pred)

        # 3) Match ndims by appending singleton dims to the end
        if y_true.ndim < y_pred.ndim:
            y_true = y_true.reshape(y_true.shape + (1,) * (y_pred.ndim - y_true.ndim))
        elif y_pred.ndim < y_true.ndim:
            y_pred = y_pred.reshape(y_pred.shape + (1,) * (y_true.ndim - y_pred.ndim))

        # 4) Broadcast to identical shapes
        try:
            y_true, y_pred = jnp.broadcast_arrays(y_true, y_pred)
        except ValueError as e:
            raise ValueError(
                f"Cannot align shapes y_true{y_true.shape} vs y_pred{y_pred.shape}"
            ) from e

        # 5) Clip predictions to avoid log(0)
        eps = 1e-7
        y_pred = jnp.clip(y_pred, eps, 1 - eps)

        # 6) Compute p_t: model’s probability of the true class
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # 7) Alpha-balancing factor
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        # 8) Focal modulating factor
        focal_weight = alpha_factor * jnp.power(1 - p_t, self.gamma)

        # 9) Standard binary crossentropy
        ce_loss = - (y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

        # 10) Apply focal weight
        return focal_weight * ce_loss




class CategoricalCrossentropy(Loss):
    """
    Computes the categorical crossentropy loss for one-hot encoded targets.

    Args:
      from_logits: Boolean indicating whether y_pred are logits. If True, applies softmax.
      label_smoothing: Float in [0, 1]. If > 0, smooth the labels.
      **kwargs: Additional keyword arguments passed to the base Loss.
    """
    def __init__(self, from_logits=False, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Apply label smoothing.
        if self.label_smoothing > 0:
            num_classes = y_true.shape[-1]
            y_true = y_true * (1 - self.label_smoothing) + (self.label_smoothing / num_classes)
        # If predictions are logits, apply softmax.
        if self.from_logits:
            y_pred = jax.nn.softmax(y_pred, axis=-1)
        epsilon = 1e-7
        y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
        # Compute categorical crossentropy.
        loss = -jnp.sum(y_true * jnp.log(y_pred), axis=-1)
        return loss

class CategoricalFocalCrossentropy(Loss):
    """
    Computes the focal variant of categorical crossentropy loss.

    Args:
      gamma: Focusing parameter that reduces the relative loss for well-classified examples.
      alpha: Balancing parameter.
      from_logits: Boolean indicating whether y_pred are logits.
      label_smoothing: Float in [0, 1] for label smoothing.
      **kwargs: Additional keyword arguments passed to the base Loss.
    """
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        if self.label_smoothing > 0:
            num_classes = y_true.shape[-1]
            y_true = y_true * (1 - self.label_smoothing) + (self.label_smoothing / num_classes)
        if self.from_logits:
            y_pred = jax.nn.softmax(y_pred, axis=-1)
        epsilon = 1e-7
        y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
        # Standard categorical crossentropy per class.
        ce_loss = -y_true * jnp.log(y_pred)
        # Apply the focal weighting.
        focal_weight = jnp.power(1 - y_pred, self.gamma)
        loss = self.alpha * focal_weight * ce_loss
        loss = jnp.sum(loss, axis=-1)
        return loss

class SparseCategoricalCrossentropy(Loss):
    """
    Computes the categorical crossentropy loss for sparse (integer) targets.

    Args:
      from_logits: Boolean indicating whether y_pred are logits.
      label_smoothing: Float in [0, 1] for label smoothing.
      **kwargs: Additional keyword arguments passed to the base Loss.
    """
    def __init__(self, from_logits=False, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Determine the number of classes from y_pred.
        num_classes = y_pred.shape[-1]
        # Convert sparse labels to one-hot vectors.
        y_true = jax.nn.one_hot(y_true, num_classes)
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + (self.label_smoothing / num_classes)
        if self.from_logits:
            y_pred = jax.nn.softmax(y_pred, axis=-1)
        epsilon = 1e-7
        y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
        loss = -jnp.sum(y_true * jnp.log(y_pred), axis=-1)
        return loss
    




class MeanSquaredError(Loss):
    """
    Computes the Mean Squared Error loss.
    """
    def call(self, y_true, y_pred):
        error = jnp.square(y_pred - y_true)
        return jnp.mean(error, axis=-1)

class MeanAbsoluteError(Loss):
    """
    Computes the Mean Absolute Error loss.
    """
    def call(self, y_true, y_pred):
        error = jnp.abs(y_pred - y_true)
        return jnp.mean(error, axis=-1)

class CosineSimilarity(Loss):
    """
    Computes the Cosine Similarity loss (as negative cosine similarity).

    Args:
      axis: The axis along which to compute the cosine similarity. Defaults to -1.
    """
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, y_true, y_pred):
        epsilon = 1e-7
        # Normalize y_true and y_pred along the specified axis.
        y_true_norm = y_true / (jnp.linalg.norm(y_true, axis=self.axis, keepdims=True) + epsilon)
        y_pred_norm = y_pred / (jnp.linalg.norm(y_pred, axis=self.axis, keepdims=True) + epsilon)
        # Compute cosine similarity and return its negative.
        cosine_sim = jnp.sum(y_true_norm * y_pred_norm, axis=self.axis)
        return -cosine_sim

class Huber(Loss):
    """
    Computes the Huber loss.

    Args:
      delta: The threshold at which to change between quadratic and linear loss. Defaults to 1.0.
    """
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, y_true, y_pred):
        error = y_pred - y_true
        abs_error = jnp.abs(error)
        quadratic = jnp.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * jnp.square(quadratic) + self.delta * linear
        return jnp.mean(loss, axis=-1)
