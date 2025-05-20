import math
from typing import Optional, Tuple

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
    High-performance, flat JAXFlow Model with full custom-call awareness, JIT/VMAP/PMAP support, and Optax integration.

    This Model:
      - Automatically discovers and flattens both “true” leaf sub-layers and any layer
        that overrides `functional_call`, so custom composites just work.
      - Provides pure-functional forward passes via `functional_call`, separate from
        the object-oriented API (`call`/`predict`), for maximal JAX speed.
      - Integrates seamlessly with Optax: handles init, update, EMA, gradient accumulation,
        and multi-device (PMAP) training.
      - Exposes familiar Keras-like methods: `build`, `compile`, `fit`, `evaluate`, `predict`.

    Parameters
    ----------
    name : Optional[str]
        Human-readable name for the model (used in summaries, checkpoints, etc.).
    trainable : bool
        If False, all sub-layers’ parameters are frozen.

    Attributes
    ----------
    layers : list[Layer]
        User-registered top-level blocks in the exact order of execution.
    _flat_layers : list[Layer]
        Internal flattened list of layers used by the pure-functional API.
    optimizer
        Optax-compatible optimizer; set in `compile()`.
    loss_fn : Callable
        Loss function accepting `(targets, predictions)`; set in `compile()`.
    metrics : list[Callable]
        Optional metric functions to compute during training.
    multi_device : bool
        Whether to shard batches across multiple devices via `pmap`.
    _opt_state, _accum_grads, _ema_params, _step
        Internal Optax training state trees and counters.
    _forward_fn, _batched_forward, _train_step, _eval_step
        JIT-compiled functions for fast forward, batched inference, train and eval steps.

    Methods
    -------
    add(layer: Layer)
        Register a new sub-layer to the top-level execution graph.
    build(input_shape: Tuple[int, ...])
        Build all sub-layers (if not already built) and JIT-compile the forward pass.
    call(inputs: Array, training: bool = False) → Array
        OO API forward pass; invokes the compiled `_forward_fn`.
    functional_call(inputs: Array, params: Mapping, training: bool = False) → Array
        Pure-functional forward pass using explicit parameter PyTree.
    compile(optimizer, loss_fn, *, metrics=None, multi_device=False)
        Initialize training state, wrap forward/eval in JIT/VMAP/PMAP, and set loss/metric fns.
    fit(X, Y, *, epochs, batch_size=32, validation_data=None, validation_split=None, verbose=1) → dict
        Train the model, returning a history of (val_)loss.
    evaluate(X, Y, *, batch_size=32, verbose=0, params=None) → float
        Compute average loss over the dataset.
    predict(X) → Array
        Run inference on a batch of inputs via the OO API.
    predict_pmap(Xs) → Array
        Shard inference across devices (requires `compile(multi_device=True)`).
    summary()
        Print a concise summary of each registered block.

    Notes
    -----
    - After training, `fit` will sync the learned parameters back into each `Layer` instance.
    - Use `get_params()` / `set_params()` to extract or inject the full PyTree of weights.
    - Designed for advanced use-cases: gradient accumulation, EMA, custom in-loop logging.
    
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, name: Optional[str] = None, trainable: bool = True):
        super().__init__(name=name, trainable=trainable)
        self.layers: list[Layer]  = []   # user-registered top blocks
        self._flat_layers: list[Layer] = []  # filled in build()

        # Training-state handles
        self.optimizer = None
        self.loss_fn   = None
        self.metrics   = []
        self.multi_device = False
        self._opt_state = None
        self._accum_grads = None
        self._ema_params  = None
        self._step        = None

    # ------------------------------------------------------------------ #
    # Layer management
    # ------------------------------------------------------------------ #

    def add(self, layer: Layer):
        if not isinstance(layer, Layer):
            raise ValueError("add() expects a Layer instance")
        self.layers.append(layer)

    # Flatten according to the custom-aware rule
    def _flatten_layers(self):
        flat = []

        def visit(L: Layer):
            subs = L._get_all_sub_layers()
            # treat as leaf if no children OR custom functional_call
            if not subs or _uses_custom_fcall(L):
                flat.append(L)
            else:
                for s in subs:
                    visit(s)

        for top in self.layers:
            visit(top)
        return flat

    # ------------------------------------------------------------------ #
    # Build & forward
    # ------------------------------------------------------------------ #

    def build(self, input_shape):
        dummy = list(input_shape)
        if not dummy or dummy[0] in (None, 0):
            dummy[0] = 1
        x = jnp.zeros(dummy, dtype=jnp.float32)

        for L in self.layers:
            if not L.built:
                L.build(x.shape)
                L.built, L.built_shape = True, (None,) + x.shape[1:]
            x = L(x, training=False)

        self._flat_layers = self._flatten_layers()
        self.built, self.built_shape = True, input_shape

        # OO forward for predict() before compile
        def _forward(inp, training: bool):
            out = inp
            for L in self.layers:
                out = L(out, training=training)
            return out

        self._forward_fn = jit(_forward, static_argnums=(1,))

    def call(self, inputs, training: bool = False):
        if not self.built:
            self.build(inputs.shape)
        return self._forward_fn(inputs, training)

    # ------------------------------------------------------------------ #
    # Param helpers (recursive gather / assign)
    # ------------------------------------------------------------------ #

    def _collect_params(self, layer: Layer):
        p = {name: var.value for name, var in layer._params.items()}
        for sub in layer._get_all_sub_layers():
            p[sub.name] = self._collect_params(sub)
        return p

    def get_params(self):
        return {
            f"layer_{i}": self._collect_params(L)
            for i, L in enumerate(self._flat_layers)
        }

    def _apply_params(self, layer: Layer, tree):
        for name, var in layer._params.items():
            var.assign(tree[name])
        for sub in layer._get_all_sub_layers():
            self._apply_params(sub, tree[sub.name])

    def set_params(self, params):
        for i, L in enumerate(self._flat_layers):
            self._apply_params(L, params[f"layer_{i}"])

    # ------------------------------------------------------------------ #
    # Pure functional forward (flat, fast)
    # ------------------------------------------------------------------ #

    def functional_call(self, inputs, params, training: bool = False):
        out = inputs
        for i, L in enumerate(self._flat_layers):
            out = L.functional_call(out, params[f"layer_{i}"], training=training)
        return out

    # ------------------------------------------------------------------ #
    # Compile
    # ------------------------------------------------------------------ #

    def compile(self, optimizer, loss_fn, *, metrics=None, multi_device=False):
        if not self.built:
            raise RuntimeError("Call build() first or run data through model.")
        self.optimizer     = optimizer
        self.loss_fn       = loss_fn
        self.metrics       = metrics or []
        self.multi_device  = multi_device

        # Batch-vectorised forward
        self._batched_forward = vmap(self._forward_fn, in_axes=(0, None))
        self._parallel_forward = (
            pmap(self._batched_forward, in_axes=(0, None))
            if multi_device else self._batched_forward
        )

        params = self.get_params()
        (self._opt_state,
         self._accum_grads,
         self._ema_params,
         self._step) = optimizer.init(params)

        self._train_step = jit(self._make_train_step())
        self._eval_step  = jit(self._make_eval_step())

    # ------------------------------------------------------------------ #
    # JIT step factories
    # ------------------------------------------------------------------ #

    def _make_train_step(self):
        def step(params, opt_state, accum, ema, count, xb, yb):
            def loss_fn(p):
                preds = self.functional_call(xb, p, training=True)
                return self.loss_fn(yb, preds)
            loss, grads = value_and_grad(loss_fn)(params)
            params, opt_state, accum, ema, count, _ = self.optimizer.update(
                params, grads, opt_state, accum, ema, count)
            return params, opt_state, accum, ema, count, loss
        return step

    def _make_eval_step(self):
        def step(params, xb, yb):
            preds = self.functional_call(xb, params, training=False)
            return self.loss_fn(yb, preds)
        return step

    # ------------------------------------------------------------------ #
    # Fit / evaluate / predict  (identical to your fast version)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_data(X, Y, val_split: float):
        Xtr, Xv, Ytr, Yv = train_test_split(X, Y, test_size=val_split, shuffle=True)
        return Xtr, Ytr, Xv, Yv

    def fit(
        self,
        X, Y,
        *, epochs: int,
        batch_size: int = 32,
        validation_data: Optional[Tuple] = None,
        validation_split: Optional[float] = None,
        verbose: int = 1,
    ):
        if validation_split is not None:
            if validation_data is not None:
                raise ValueError("Use either validation_data or validation_split.")
            if not (0.0 < validation_split < 1.0):
                raise ValueError("validation_split must be in (0,1)")
            X, Y, Xv, Yv = self._split_data(X, Y, validation_split)
            validation_data = (Xv, Yv)

        params   = self.get_params()
        opt_state, accum, ema, step = (
            self._opt_state, self._accum_grads, self._ema_params, self._step)

        n_samples       = X.shape[0]
        steps_per_epoch = max(1, math.ceil(n_samples / batch_size))

        history = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []

        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"Epoch {epoch}/{epochs}")
            running = 0.0
            bar = (
                tqdm(total=steps_per_epoch, desc="Training", unit="batch",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] • {postfix}")
                if verbose else None
            )

            for step_idx in range(steps_per_epoch):
                s = step_idx * batch_size
                e = min(s + batch_size, n_samples)
                xb, yb = X[s:e], Y[s:e]

                params, opt_state, accum, ema, step, loss_val = self._train_step(
                    params, opt_state, accum, ema, step, xb, yb)

                running += float(loss_val)
                if verbose:
                    bar.update(1)
                    bar.set_postfix({"loss": f"{running/(step_idx+1):.4f}"})

            if verbose:
                bar.close()

            avg_loss = running / steps_per_epoch
            history["loss"].append(avg_loss)

            if validation_data is not None:
                Xv, Yv = validation_data
                val_loss = self.evaluate(
                    Xv, Yv, batch_size=batch_size, verbose=0, params=params)
                history["val_loss"].append(val_loss)
                if verbose:
                    print(f"loss: {avg_loss:.4f} — val_loss: {val_loss:.4f}")

        # sync trained weights back into live objects
        self.set_params(params)
        self._opt_state, self._accum_grads, self._ema_params, self._step = (
            opt_state, accum, ema, step)
        return history

    def evaluate(
        self,
        X, Y,
        *, batch_size: int = 32,
        verbose: int = 0,
        params: Optional[dict] = None,
    ) -> float:
        if params is None:
            params = self.get_params()

        n     = X.shape[0]
        steps = max(1, math.ceil(n / batch_size))
        total = 0.0
        bar = (
            tqdm(total=steps, desc="Evaluating", unit="batch",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] • {postfix}")
            if verbose else None
        )
        for i in range(steps):
            xb = X[i*batch_size:(i+1)*batch_size]
            yb = Y[i*batch_size:(i+1)*batch_size]
            total += float(self._eval_step(params, xb, yb))
            if verbose:
                bar.update(1)
                bar.set_postfix({"loss": f"{total/(i+1):.4f}"})
        if verbose:
            bar.close()
        return total / steps

    # ------------------------------------------------------------------ #
    # Prediction helpers
    # ------------------------------------------------------------------ #

    def predict(self, X):
        if not hasattr(self, "_batched_forward"):
            return vmap(lambda x: self.call(x, training=False))(X)
        return self.call(X, training=False)

    def predict_pmap(self, Xs):
        if not self.multi_device:
            raise RuntimeError("Compile with multi_device=True")
        devices = jax.local_devices()
        params  = self.get_params()
        pr      = jax.device_put_replicated(params, devices)
        return pmap(self._batched_forward, in_axes=(0, None))(Xs, False)
    

    def __setattr__(self, name, value):
        """Auto-register every Layer assigned as an attribute.

        Works only *after* self.layers has been created (i.e. after __init__).  
        Sequential-style `self.add(layer)` still works; duplicates are ignored.
        """
        super().__setattr__(name, value)                     # normal assignment first
        if isinstance(value, Layer) and 'layers' in self.__dict__:
            if value not in self.layers:
                self.layers.append(value)

    # ------------------------------------------------------------------ #
    # Misc.
    # ------------------------------------------------------------------ #

    def summary(self):
        print(f"Model '{self.name}' summary:")
        for i, L in enumerate(self.layers):
            print(f"  Block {i}: {L}")

    def __repr__(self):
        return f"<Model {self.name}, built={self.built}, leafs={len(self._flat_layers)}>"
