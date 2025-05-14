import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.tree_util import tree_map
from jaxflow.layers.layer import Layer

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

    def call(self, inputs, training=False, mask=None):
        if self.layers:
            x = inputs
            for layer in self.layers:
                x = layer(x, training=training, mask=mask)
            return x
        raise NotImplementedError("Custom model must override call() or add layers via add().")

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
        self.compiled = True

    def train_step(self, x, y, sample_weight=None, mask=None):
        """
        Execute one training step using the custom optimizer.
        This stateful version sets model parameters via side effects.
        """
        loss_val,params, grads = self.compute_gradients(x, y, sample_weight=sample_weight, mask=mask)
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

    def compute_gradients(self, x, y, loss_fn=None, sample_weight=None, mask=None):
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

        params = self.get_params()
        loss_val, grads = jax.value_and_grad(loss_wrapper)(params, x, y, sample_weight, mask)
        return loss_val,params , grads

    def evaluate(self, x, y, batch_size=32, sample_weight=None, mask=None):
        num_samples = x.shape[0]
        steps = num_samples // batch_size
        total_loss = 0.0
        for step in range(steps):
            batch_x = x[step * batch_size:(step + 1) * batch_size]
            batch_y = y[step * batch_size:(step + 1) * batch_size]
            preds = self(batch_x, training=False)
            loss_val = self.loss_fn(batch_y, preds, sample_weight=sample_weight, mask=mask)
            total_loss += loss_val
        return total_loss / steps

    def predict(self, x, batch_size=32):
        return self(x, training=False)

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1, sample_weight=None, mask=None, validation_data=None):
        num_samples = x.shape[0]
        steps_per_epoch = num_samples // batch_size
        for epoch in range(epochs):
            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                batch_x = x[step * batch_size:(step + 1) * batch_size]
                batch_y = y[step * batch_size:(step + 1) * batch_size]
                loss_val = self.train_step(batch_x, batch_y, sample_weight=sample_weight, mask=mask)
                epoch_loss += loss_val
            epoch_loss /= steps_per_epoch
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        return

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
