import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap, pmap
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from jaxflow.layers.layer import Layer

class Model(Layer):
    """
    High-performance Model for JAXFlow with JIT, VMAP, PMAP support,
    integrated with a pure-functional BaseOptimizer (Optax wrapper).
    Automatically discovers both explicit and attribute-defined sub-layers.
    """

    def __init__(self, name=None, trainable=True):
        super().__init__(name=name, trainable=trainable)
        # explicitly registered layers
        self.layers = []
        # optimizer and training state
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []
        self.multi_device = False
        self._opt_state = None
        self._accum_grads = None
        self._ema_params = None
        self._step = None

    def add(self, layer):
        """
        Explicitly add a Layer to the model.
        """
        if not isinstance(layer, Layer):
            raise ValueError("Added object must be a Layer instance")
        self.layers.append(layer)

    def _get_all_sub_layers(self):
        explicit = list(self.layers)
        inherited = super()._get_all_sub_layers()
        return explicit + [l for l in inherited if l not in explicit]

    def build(self, input_shape):
        # build all layers via a dummy batch
        bs = list(input_shape)
        if not bs or bs[0] in (None, 0): bs[0] = 1
        x = jnp.zeros(bs, dtype=jnp.float32)
        subs = self._get_all_sub_layers()
        for L in subs:
            if not L.built:
                L.build(x.shape)
                L.built = True
                L.built_shape = (None,) + x.shape[1:]
            x = L(x, training=False)
        self.built = True
        self.built_shape = input_shape
        # fused forward with Python loop
        def _forward(x, training: bool):
            out = x
            for layer in subs:
                out = layer(out, training=training)
            return out
        self._forward_fn = jit(_forward, static_argnums=(1,))

    def call(self, inputs, training=False):
        if not self.built:
            self.build(inputs.shape)
        return self._forward_fn(inputs, training)

    def functional_call(self, inputs, params, training=False):
        out = inputs
        subs = self._get_all_sub_layers()
        for i, L in enumerate(subs):
            layer_params = params[f"layer_{i}"]
            out = L.functional_call(out, layer_params, training=training)
        return out

    def get_params(self):
        subs = self._get_all_sub_layers()
        return {f"layer_{i}": {n: v.value for n, v in L._params.items()}
                for i, L in enumerate(subs)}

    def set_params(self, params):
        subs = self._get_all_sub_layers()
        for i, L in enumerate(subs):
            for n, v in L._params.items():
                v.assign(params[f"layer_{i}"][n])

    def compile(self, optimizer, loss_fn, metrics=None, multi_device=False):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.multi_device = multi_device
        if not self.built:
            raise RuntimeError("Build the model before compile.")
        # batched and multi-device forward
        self._batched_forward = vmap(self._forward_fn, in_axes=(0, None))
        self._parallel_forward = (
            pmap(self._batched_forward, in_axes=(0, None))
            if multi_device else self._batched_forward
        )
        # init optimizer state
        params = self.get_params()
        self._opt_state, self._accum_grads, self._ema_params, self._step = \
            optimizer.init(params)
        # jit'd steps
        self._train_step = jit(self._make_train_step())
        self._eval_step  = jit(self._make_eval_step())

    @staticmethod
    def _split_data(X, Y, validation_split):
        """
        Split X, Y into training and validation sets, using sklearn.shuffle.
        """
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=validation_split, shuffle=True
        )
        return X_train, Y_train, X_val, Y_val

    def _make_train_step(self):
        def train_step(params, opt_state, accum, ema, step, xb, yb):
            def lw(p):
                preds = self.functional_call(xb, p, training=True)
                return self.loss_fn(yb, preds)
            loss, grads = value_and_grad(lw)(params)
            new_p, new_s, new_a, new_e, new_step, _ = \
                self.optimizer.update(params, grads, opt_state, accum, ema, step)
            return new_p, new_s, new_a, new_e, new_step, loss
        return train_step

    def _make_eval_step(self):
        def eval_step(params, xb, yb):
            preds = self.functional_call(xb, params, training=False)
            return self.loss_fn(yb, preds)
        return eval_step

    def fit(self, X, Y,
            epochs,
            batch_size=32,
            validation_data=None,
            validation_split=None,
            verbose=1):
        """
        Train the model with optional validation split or data.
        """
        # If using validation_split, randomize and split
        if validation_split is not None:
            if validation_data is not None:
                raise ValueError("Cannot specify both validation_data and validation_split")
            if not (0.0 < validation_split < 1.0):
                raise ValueError("validation_split must be in (0,1)")
            X_train, Y_train, X_val, Y_val = self._split_data(X, Y, validation_split)
            validation_data = (X_val, Y_val)
            X, Y = X_train, Y_train
        # prepare training
        n = X.shape[0]
        params = self.get_params()
        os, ag, ep, st = self._opt_state, self._accum_grads, self._ema_params, self._step
        steps = max(1, math.ceil(n / batch_size))
        history = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []
        for e in range(1, epochs + 1):
            print(f"Epoch {e}/{epochs}")
            total_loss = 0.0
            bar = tqdm(total=steps, desc="Training", unit="batch",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] • {postfix}") if verbose else None
            for i in range(steps):
                xb = X[i*batch_size:(i+1)*batch_size]
                yb = Y[i*batch_size:(i+1)*batch_size]
                params, os, ag, ep, st, loss = self._train_step(params, os, ag, ep, st, xb, yb)
                total_loss += float(loss)
                if verbose:
                    bar.update(1); bar.set_postfix({"loss": f"{(total_loss/(i+1)):.4f}"})
            if verbose: bar.close()
            avg_loss = total_loss / steps
            history["loss"].append(avg_loss)
            if validation_data is not None:
                Xv, Yv = validation_data
                val_loss = self.evaluate(Xv, Yv, batch_size=batch_size, verbose=0)
                history["val_loss"].append(val_loss)
                print(f"loss: {avg_loss:.4f} — val_loss: {val_loss:.4f}")
        # restore state
        self.set_params(params)
        self._opt_state, self._accum_grads, self._ema_params, self._step = os, ag, ep, st
        return history

    def evaluate(self, X, Y, batch_size=32, verbose=0):
        params = self.get_params()
        n = X.shape[0]; steps = max(1, math.ceil(n/batch_size))
        total_loss = 0.0
        if verbose:
            bar = tqdm(total=steps, desc="Evaluating", unit="batch",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] • {postfix}")
        for i in range(steps):
            xb = X[i*batch_size:(i+1)*batch_size]
            yb = Y[i*batch_size:(i+1)*batch_size]
            loss = self._eval_step(params, xb, yb)
            total_loss += float(loss)
            if verbose:
                bar.update(1); bar.set_postfix({"loss": f"{(total_loss/(i+1)):.4f}"})
        if verbose: bar.close()
        print(f"Test Loss: {total_loss/steps:.4f}")
        return total_loss/steps

    def predict(self, X):
        return self._batched_forward(X, False)

    def predict_pmap(self, Xs):
        if not self.multi_device:
            raise RuntimeError("Compile with multi_device=True")
        devices = jax.local_devices()
        params = self.get_params()
        pr = jax.device_put_replicated(params, devices)
        return pmap(self._batched_forward, in_axes=(0, None))(Xs, False)

    def summary(self):
        print(f"Model '{self.name}' summary:")
        for i, L in enumerate(self._get_all_sub_layers()):
            print(f"  Layer {i}: {L}")

    def __repr__(self):
        return f"<Model {self.name}, built={self.built}, layers={len(self._get_all_sub_layers())}>"