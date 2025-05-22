import math
from typing import Optional, Tuple, Mapping, Any

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap, pmap
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from jaxflow.layers.layer import Layer

# --------------------------------------------------------------------- #
# Helper: does a layer override functional_call?
# --------------------------------------------------------------------- #
def _uses_custom_fcall(layer: Layer) -> bool:
    """Return True if *layer* implements its own functional_call."""
    return layer.functional_call.__func__ is not Layer.functional_call

class Model(Layer):
    """
    High-performance, flat JAXFlow Model with custom-call awareness, JIT/VMAP/PMAP support, and Optax integration.

    ...
    """
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, name: Optional[str] = None, trainable: bool = True):
        super().__init__(name=name, trainable=trainable)
        self.layers: list[Layer] = []
        self._flat_layers: list[Layer] = []
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []
        self.multi_device = False
        self._opt_state = None
        self._accum_grads = None
        self._ema_params = None
        self._step = None

    # ------------------------------------------------------------------ #
    # Layer management
    # ------------------------------------------------------------------ #
    def add(self, layer: Layer):
        """Register a new sub-layer to the execution graph."""
        if not isinstance(layer, Layer):
            raise ValueError("add() expects a Layer instance")
        self.layers.append(layer)

    def __setattr__(self, key: str, value: Any):
        """
        Auto-register any Layer assigned as an attribute into both
        `_sub_layers` (via base Layer) and `layers` (for execution).
        """
        super().__setattr__(key, value)
        if isinstance(value, Layer) and 'layers' in self.__dict__:
            if value not in self.layers:
                self.layers.append(value)

    # ------------------------------------------------------------------ #
    # Flattening logic (top-level only)
    # ------------------------------------------------------------------ #
    def _flatten_layers(self) -> list[Layer]:
        """Return top-level layers as leaves for the functional API."""
        return list(self.layers)

    # ------------------------------------------------------------------ #
    # Build & forward
    # ------------------------------------------------------------------ #
    def build(self, input_shape: Tuple[int, ...]):
        """Build all sub-layers and compile the OO forward function."""
        dummy = list(input_shape)
        if not dummy or dummy[0] in (None, 0):
            dummy[0] = 1
        x = jnp.zeros(dummy, dtype=jnp.float32)
        for L in self.layers:
            if not L.built:
                L.build(x.shape)
                L.built = True
                L.built_shape = (None,) + x.shape[1:]
            x = L(x, training=False)
        self._flat_layers = self._flatten_layers()
        self.built = True
        self.built_shape = input_shape

        def _forward(inp: jnp.ndarray, training: bool):
            out = inp
            for L in self.layers:
                out = L(out, training=training)
            return out
        self._forward_fn = jit(_forward, static_argnums=(1,))

    def call(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """OO API forward pass using the compiled forward function."""
        if not self.built:
            self.build(inputs.shape)
        return self._forward_fn(inputs, training)

    # ------------------------------------------------------------------ #
    # Pure functional forward
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def functional_call(
        self,
        inputs: jnp.ndarray,
        params: Mapping[str, Any],
        training: bool = False
    ) -> jnp.ndarray:
        """
        Pure-functional forward: temporarily inject a full parameter PyTree,
        run the OO `call` capturing all operations, then restore original parameters.
        """
        # Inject new parameters
        self.set_params(params)

        # Use the OO API; all ops in `call` are recorded
        out = self.call(inputs, training=training)

        return out

        # ------------------------------------------------------------------ #
    # Param helpers (recursive gather / assign)
    # ------------------------------------------------------------------ #
    def _collect_params(self, layer: Layer) -> dict:
        """
        Recursively collect only trainable variables from each layer.
        If a sub-layer has no trainable variables, it will be omitted.
        """
        # 1) Gather trainable vars at this level
        p = {
            name: var.value
            for name, var in layer._params.items()
            if var.trainable
        }

        # 2) Recurse into sub-layers
        for sub in layer._get_all_sub_layers():
            sub_p = self._collect_params(sub)
            # Only include this sub-layer if it had any trainable vars
            if sub_p:
                p[sub.name] = sub_p

        return p

    def get_params(self) -> dict:
        """Extract a PyTree of all parameters for the functional API."""
        return {f"layer_{i}": self._collect_params(L)
                for i, L in enumerate(self._flat_layers)}

    def _apply_params(self, layer: Layer, tree: dict):
        """Recursively assign raw arrays from a nested dict back into Variables."""

        for name, var in layer._params.items():
            var.assign(tree[name])
        for sub in layer._get_all_sub_layers():
            self._apply_params(sub, tree[sub.name])

    def set_params(self, params: dict):
        """Inject a PyTree of parameters back into the model's layers."""
        for i, L in enumerate(self._flat_layers):
            # check if layer empty no params 
            if params[f"layer_{i}"]:
                self._apply_params(L, params[f"layer_{i}"])

    # ------------------------------------------------------------------ #
    # Compile
    # ------------------------------------------------------------------ #
    def compile(self,
                optimizer,
                loss_fn,
                *, metrics=None,
                multi_device: bool = False):
        """Initialize optimizer state, wrap forward and train/eval with JIT/VMAP/PMAP."""
        if not self.built:
            raise RuntimeError("Call build() before compile().")
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.multi_device = multi_device

        self._batched_forward = vmap(self._forward_fn, in_axes=(0, None))
        self._parallel_forward = (
            pmap(self._batched_forward, in_axes=(0, None)) if multi_device
            else self._batched_forward
        )

        params = self.get_params()
        self._opt_state, self._accum_grads, self._ema_params, self._step = (
            optimizer.init(params)
        )

        self._train_step = jit(self._make_train_step())
        self._eval_step  = jit(self._make_eval_step())

    # ------------------------------------------------------------------ #
    # JIT step factories
    # ------------------------------------------------------------------ #
    def _make_train_step(self):
        """Factory for a single JIT-compiled training step."""
        def step(params, opt_state, accum, ema, count, xb, yb):
            def _loss(p):
                preds = self.functional_call(xb, p, training=True)
                return self.loss_fn(yb, preds)
            loss, grads = value_and_grad(_loss)(params)
            params, opt_state, accum, ema, count, _ = (
                self.optimizer.update(params, grads, opt_state, accum, ema, count)
            )
            return params, opt_state, accum, ema, count, loss
        return step

    def _make_eval_step(self):
        """Factory for a single JIT-compiled evaluation step."""
        def step(params, xb, yb):
            preds = self.functional_call(xb, params, training=False)
            return self.loss_fn(yb, preds)
        return step

    # ------------------------------------------------------------------ #
    # Fit / evaluate / predict
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_data(X, Y, val_split: float):
        return train_test_split(X, Y, test_size=val_split, shuffle=True)

    def fit(
        self,
        X: Any,
        Y: Any,
        *, epochs: int,
        batch_size: int = 32,
        validation_data: Optional[Tuple] = None,
        validation_split: Optional[float] = None,
        verbose: int = 1
    ) -> dict:
        """Train the model for a fixed number of epochs."""
        if validation_split is not None:
            if validation_data is not None:
                raise ValueError("Use either validation_split or validation_data.")
            X, Xv, Y, Yv = self._split_data(X, Y, validation_split)
            validation_data = (Xv, Yv)

        params, opt_state, accum, ema, step = (
            self.get_params(), self._opt_state,
            self._accum_grads, self._ema_params, self._step
        )
        n_samples = X.shape[0]
        steps_per_epoch = max(1, math.ceil(n_samples / batch_size))
        history = {"loss": []}
        if validation_data:
            history["val_loss"] = []

        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"Epoch {epoch}/{epochs}")
            running = 0.0
            bar = tqdm(
                total=steps_per_epoch,
                desc="Training",
                unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] • {postfix}"
            ) if verbose else None

            for i in range(steps_per_epoch):
                s, e = i*batch_size, min((i+1)*batch_size, n_samples)
                xb, yb = X[s:e], Y[s:e]
                params, opt_state, accum, ema, step, loss = (
                    self._train_step(params, opt_state, accum, ema, step, xb, yb)
                )
                running += float(loss)
                if verbose:
                    bar.update(1)
                    bar.set_postfix({"loss": f"{running/(i+1):.4f}"})
            if verbose: bar.close()
            avg = running / steps_per_epoch
            history["loss"].append(avg)
            if validation_data:
                Xv, Yv = validation_data
                val = self.evaluate(Xv, Yv, batch_size=batch_size, verbose=0, params=params)
                history["val_loss"].append(val)
                if verbose:
                    print(f"loss: {avg:.4f} — val_loss: {val:.4f}")

            self.set_params(params)
        self._opt_state, self._accum_grads, self._ema_params, self._step = (
            opt_state, accum, ema, step
        )
        return history

    def evaluate(
        self,
        X: Any,
        Y: Any,
        *, batch_size: int = 32,
        verbose: int = 0,
        params: Optional[dict] = None
    ) -> float:
        """Compute loss over a dataset without updating parameters."""
        params = params or self.get_params()
        n, steps = X.shape[0], max(1, math.ceil(X.shape[0]/batch_size))
        total = 0.0
        bar = tqdm(
            total=steps,
            desc="Evaluating",
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] • {postfix}"
        ) if verbose else None
        for i in range(steps):
            xb, yb = X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size]
            total += float(self._eval_step(params, xb, yb))
            if verbose:
                bar.update(1)
                bar.set_postfix({"loss": f"{total/(i+1):.4f}"})
        if verbose: bar.close()
        return total / steps

    # ------------------------------------------------------------------ #
    # Prediction helpers
    # ------------------------------------------------------------------ #
    def predict(self, X: Any) -> Any:
        """Run inference on a batch via the OO API."""
        if not hasattr(self, '_batched_forward'):
            return vmap(lambda x: self.call(x, training=False))(X)
        return self.call(X, training=False)

    def predict_pmap(self, Xs: Any) -> Any:
        """Shard inference across devices (requires multi_device=True)."""
        if not self.multi_device:
            raise RuntimeError("Compile with multi_device=True")
        devices = jax.local_devices()
        params = self.get_params()
        pr = jax.device_put_replicated(params, devices)
        return pmap(self._batched_forward, in_axes=(0, None))(Xs, False)

    # ------------------------------------------------------------------ #
    # Misc.
    # ------------------------------------------------------------------ #
    def summary(self):
        """Print a concise summary of each registered block."""
        print(f"Model '{self.name}' summary:")
        for i, L in enumerate(self.layers):
            print(f"  Block {i}: {L}")

    def __repr__(self) -> str:
        """Brief representation."""
        return f"<Model {self.name}, built={self.built}, leaves={len(self._flat_layers)}>"
