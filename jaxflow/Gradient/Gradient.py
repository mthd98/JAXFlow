import jax
import jax.numpy as jnp
from jax import grad
# -------------------------------
# Custom GradientTape with automatic watching.
# -------------------------------
class GradientTape:
    def __init__(self, persistent=False):
        self.persistent = persistent
        self._watched = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if not self.persistent:
            self._clear_record()

    def _clear_record(self):
        self.f = None
        self.args = None
        self.kwargs = None
        self.value = None
        self._watched = None

    def record(self, f, *args, **kwargs):
        # Automatically watch the first argument (typically parameters)
        if self._watched is None:
            self._watched = [args[0]]
        self.f = f
        self.args = args
        self.kwargs = kwargs
        self.value = f(*args, **kwargs)
        return self.value

    def gradient(self, order=1):
        if self.f is None:
            raise ValueError("No function recorded. Please call record() first.")

        def loss_fn(param):
            new_args = (param,) + self.args[1:]
            return self.f(*new_args, **self.kwargs)

        grad_fn = loss_fn
        for _ in range(order):
            grad_fn = jax.grad(grad_fn)
        return grad_fn(self._watched[0])

