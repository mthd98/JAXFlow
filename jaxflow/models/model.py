import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.tree_util import tree_map
from jaxflow.layers.layer import Layer
from tqdm import tqdm
import math

class Model(Layer):
    """
    Enhanced Model class for jaxflow that supports custom subclass models.

    This class gathers sub-layers both from:
      - Layers explicitly added via add()
      - Instance attributes that are Layer objects
    """
    def __init__(self, name=None, trainable=True):
        super().__init__(name=name, trainable=trainable)
        self.layers = []         # Explicitly registered layers (order matters)
        self.compiled = False    # Whether compile() has been called
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []

    def add(self, layer):
        """Explicitly add a layer to the model."""
        if not isinstance(layer, Layer):
            raise ValueError("Added object must be an instance of Layer")
        self.layers.append(layer)

    def _get_all_sub_layers(self):
        explicit_layers = list(self.layers)
        inherited_layers = super()._get_all_sub_layers()
        all_layers = explicit_layers.copy()
        for layer in inherited_layers:
            if layer not in all_layers:
                all_layers.append(layer)
        return all_layers

    def build(self, input_shape):
        """
        Build the model by running a dummy forward pass.
        """
        dummy_shape = list(input_shape)
        if dummy_shape[0] is None:
            dummy_shape[0] = 1
        x = jnp.zeros(tuple(dummy_shape))
        if self.layers:
            for layer in self.layers:
                layer.build(x.shape)
                x = layer(x, training=False)
                layer.built_shape = (None,) + x.shape[1:]
        else:
            x = self.call(x, training=False)
        self.built = True
        self.built_shape = input_shape

        def _scanned_forward(x, training, mask):
            # body_fn: (carry, layer) -> (new_carry, nothing)
            def body_fn(carry, layer):
                out = layer(carry, training=training, mask=mask)
                return out, None
            # lax.scan over the tuple of layers:
            final, _ = jax.lax.scan(body_fn, x, tuple(self.layers))
            return final

        # 2) JIT it, marking the booleans as static so they don’t get traced every call:
        #    static_argnums=(1,2) means “treat training and mask as compile-time constants”
        self._forward_fn = jax.jit(_scanned_forward, static_argnums=(1,2))

    def call(self, inputs, training=False, mask=None):        

        return self._forward_fn(inputs, training, mask)

    def get_params(self):
        """
        Gather model parameters from this layer and all sub-layers.
        The parameters are organized in a dictionary with keys "layer_0", "layer_1", etc.
        """
        params = {}
        for i, layer in enumerate(self._get_all_sub_layers()):
            layer_params = {name: var.value for name, var in layer._params.items()}
            params[f"layer_{i}"] = layer_params
        return params

    def set_params(self, params):
        """
        Update the internal state of the model using the provided parameter tree.
        """
        for i, layer in enumerate(self._get_all_sub_layers()):
            layer_params = params[f"layer_{i}"]
            for name, var in layer._params.items():
                var.assign(layer_params[name])

    def compile(self, optimizer, loss, metrics=[]):
        """
        Configure the model for training using a custom optimizer.

        Here, optimizer is expected to be an instance of your custom optimizer,
        which implements init(params) and apply_gradients(params, grads).
        """
        self.optimizer = optimizer
        self.loss_fn = loss  # loss must be a callable
        self.metrics = metrics
        params = self.get_params()
        self.optimizer.init(params)


        self.compute_gradients_jit = jax.jit(self.compute_gradients)

        


        self.compiled = True




    def train_step(self, x, y, sample_weight=None, mask=None):
        """
        Execute one training step using the custom optimizer.
        This stateful version sets model parameters via side effects.
        """
        params = self.get_params()
        loss_val,params, grads = self.compute_gradients_jit(x, y, params,sample_weight=sample_weight, mask=mask)
        self.update_parameters(params, grads, optimizer=self.optimizer)
        return loss_val

    def functional_call(self, x, params, training=False, mask=None):
        """
        Compute the forward pass of the model using the given parameter tree.
        This method does not modify any state.

        For models with sub-layers, we assume that each sub-layer also implements
        a functional_call method and that the parameter tree is organized as a
        dictionary with keys "layer_0", "layer_1", etc.
        """
        if self.layers:
            out = x
            for i, layer in enumerate(self.layers):
                # Get parameters for each sub-layer from the parameter tree.
                layer_params = params.get(f"layer_{i}")
                if layer_params is None:
                    raise ValueError(f"Parameters for layer_{i} not found in the parameter tree.")
                # Assume each layer has a functional_call method.
                out = layer.functional_call(out, layer_params, training=training, mask=mask)
            return out
        else:
            # If no sub-layers, we use the normal call (assuming it's pure enough).
            return self.call(x, training=training, mask=mask)

    def update_parameters(self, params,grads, optimizer=None):
        """
        Update the model parameters using the provided gradients and optimizer.
        This method calls the optimizer's apply_gradients to produce new parameters,
        and then updates the model's internal state.
        """
        if optimizer is None:
            if self.optimizer is None:
                raise ValueError("No optimizer provided and the model has no default optimizer set.")
            optimizer = self.optimizer

        #params = self.get_params()
        new_params = optimizer.apply_gradients(params, grads)
        self.set_params(new_params)
        return new_params

    def compute_gradients(self, x, y, params,loss_fn=None, sample_weight=None, mask=None):
        """
        Compute and return the loss and gradients of the model with respect to its parameters,
        using a pure function that does not mutate the model state.

        Args:
            x: Input batch.
            y: Target batch.
            loss_fn: Optional loss function to override the model's loss_fn.
            sample_weight: Optional sample weights.
            mask: Optional mask for inputs.

        Returns:
            loss_val: The computed loss value.
            grads: A PyTree of gradients matching the structure of model parameters.
        """
        effective_loss_fn = loss_fn if loss_fn is not None else self.loss_fn

        def loss_wrapper(params, x, y, sample_weight, mask):
            self.set_params(params)
            preds = self(x, training=True)
            return effective_loss_fn(y, preds, sample_weight=sample_weight, mask=mask)

        loss_val, grads = jax.value_and_grad(loss_wrapper)(params, x, y, sample_weight, mask)
        return loss_val,params , grads

    
    def train_epoch(self, x, y,
                    batch_size=32,
                    sample_weight=None,
                    mask=None,
                    verbose=1):
        """Run a single training epoch and return a dict of average metrics."""
        num_samples     = x.shape[0]
        steps_per_epoch = max(1, math.ceil(num_samples / batch_size))

        running_loss    = 0.0
        running_metrics = {m.name: 0.0 for m in self.metrics}

        bar = None
        if verbose:
            bar = tqdm(total=steps_per_epoch,
                       desc="Training",
                       unit="step",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] • {postfix}")

        for step in range(steps_per_epoch):
            start = step * batch_size
            end   = min(start + batch_size, num_samples)
            xb, yb = x[start:end], y[start:end]

            # forward + backward + step
            loss_val = self.train_step(xb, yb,
                                       sample_weight=sample_weight,
                                       mask=mask)
            running_loss += float(loss_val)

            # compute any additional metrics
            preds = self.predict(xb)
            for m in self.metrics:
                running_metrics[m.name] += float(m(yb, preds))

            if verbose:
                avg_loss = running_loss / (step + 1)
                postfix = {"loss": f"{avg_loss:.4f}"}
                for name in running_metrics:
                    postfix[name] = f"{running_metrics[name]/(step+1):.4f}"
                
                bar.update(1)
                bar.set_postfix(postfix)

        if verbose:
            bar.close()

        # assemble averages
        results = {"loss": running_loss / steps_per_epoch}
        results.update({name: running_metrics[name] / steps_per_epoch
                        for name in running_metrics})
        return results

    def evaluate(self, x, y,
                 batch_size=32,
                 sample_weight=None,
                 mask=None,
                 verbose=0):
        """
        Compute loss and metrics over the dataset.
        If verbose=1, show a tqdm bar with per‐batch loss.
        """
        num_samples = x.shape[0]
        steps       = max(1, math.ceil(num_samples / batch_size))

        total_loss    = 0.0
        total_metrics = {m.name: 0.0 for m in self.metrics}

        bar = None
        if verbose:
            bar = tqdm(
                total=steps,
                desc="Evaluating",
                unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] • {postfix}"
            )

        for step in range(steps):
            start = step * batch_size
            end   = min(start + batch_size, num_samples)
            xb, yb = x[start:end], y[start:end]

            preds    = self.predict(xb)
            loss_val = self.loss_fn(
                yb, preds,
                sample_weight=sample_weight,
                mask=mask
            )
            total_loss += float(loss_val)

            # accumulate metrics
            batch_postfix = {"loss": f"{loss_val:.4f}"}
            for m in self.metrics:
                m_val = float(m(yb, preds))
                total_metrics[m.name] += m_val
                batch_postfix[m.name] = f"{m_val:.4f}"

            if verbose:
                bar.update(1)
                bar.set_postfix(batch_postfix)

        if verbose:
            bar.close()

        # compute averages
        results = {"loss": total_loss / steps}
        results.update({name: total_metrics[name] / steps
                        for name in total_metrics})
        return results

    def fit(self, x, y,
            epochs=1,
            batch_size=32,
            sample_weight=None,
            mask=None,
            validation_data=None,
            verbose=1):
        history = {"loss": [], **{m.name: [] for m in self.metrics}}
        if validation_data:
            history.update({f"val_{k}": [] for k in history})

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            train_res = self.train_epoch(x, y,
                                         batch_size=batch_size,
                                         sample_weight=sample_weight,
                                         mask=mask,
                                         verbose=verbose)
            # record training
            for k, v in train_res.items():
                history[k].append(v)

            # validation
            if validation_data:
                x_val, y_val = validation_data
                val_res = self.evaluate(x_val, y_val,
                                        batch_size=batch_size,
                                        sample_weight=sample_weight,
                                        mask=mask,
                                        verbose=verbose)
                for k, v in val_res.items():
                    history[f"val_{k}"].append(v)

                # print a combined summary line
                metrics_str = " - ".join([
                    f"{k}: {train_res[k]:.4f}" for k in train_res
                ] + [
                    f"val_{k}: {val_res[k]:.4f}" for k in val_res
                ])
                print(f"{metrics_str}")
            else:
                # no validation, just print training metrics
                metrics_str = " - ".join([
                    f"{k}: {train_res[k]:.4f}" for k in train_res
                ])
                print(metrics_str)

        return history
           
    

    def predict(self, x, mask=None):
        return self(x, training=False, mask=mask)



    def summary(self):
        print("Model Summary:")
        for i, layer in enumerate(self._get_all_sub_layers()):
            print(f"  Layer {i}: {layer}")

    def get_config(self):
        return {"name": self.name, "trainable": self.trainable,
                "built": self.built, "built_shape": self.built_shape,
                "param_names": list(self._params.keys()),
                "sub_layers": list(self._sub_layers.keys())}

    def __repr__(self):
        config = self.get_config()
        return (f"<Model {config['name']}, built={config['built']}, "
                f"trainable={config['trainable']}, param_count={len(config['param_names'])}>")
