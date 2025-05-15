import jax
import jax.numpy as jnp
import abc
import inspect
from jaxflow.core.variable import Variable

class Layer(abc.ABC):
    """
    Base class for layers and composite blocks in jaxflow.

    Subclasses should override:
      - build(input_shape): to create variables and sub-layers.
      - call(inputs, training=False, mask=None): to define the forward pass.

    This class supports composite blocks by automatically gathering sub-layers
    both from explicit registration (add_sub_layer) and instance attributes.

    Now supports an optional mask argument for the passed data.
    """
    def __init__(self, name=None, trainable=True):
        self.name = name if name is not None else self.__class__.__name__
        self.trainable = trainable
        self._params = {}         # Dictionary for parameters (e.g., weights, biases)
        self._sub_layers = {}     # Explicitly registered sub-layers
        self.built = False
        self.built_shape = None

    @abc.abstractmethod
    def build(self, input_shape):
        """
        Create variables and sub-layers based on the input shape.
        Must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def call(self, inputs, training=False, mask=None):
        """
        Define the forward pass.
        Must be implemented by subclasses.

        The mask argument is provided to allow for selective propagation of information.
        """
        pass
    
    def __call__(self, inputs, training=False, mask=None):
        if not self.built:
            inferred_shape = self._infer_input_shape(inputs)
            self.build(inferred_shape)
            self.built = True
            self.built_shape = inferred_shape
        # Check if the subclass's call method accepts a 'mask' parameter.
        sig = inspect.signature(self.call)
        if "mask" in sig.parameters:
            output_mask = self.compute_mask(inputs, mask)
            return self.call(inputs, training=training, mask=output_mask)
        else:
            return self.call(inputs, training=training)

    @staticmethod
    def _infer_input_shape(inputs):
        if isinstance(inputs, (list, tuple)):
            return [inp.shape for inp in inputs]
        else:
            return inputs.shape

    def compute_mask(self, inputs, mask):
        """
        Computes an output mask based on the input mask.

        By default, it simply passes the mask along. Subclasses can override this
        if they modify the input shape or want to change the mask.
        """
        return mask

    def functional_call(self, inputs, params, training=False, mask=None):
        """
        Default functional forward pass.

        This method is intended to be a pure (side-effect free) version of the forward pass.
        By default, it simply calls the layer's regular call() method.
        Subclasses can override this method to explicitly use the provided parameter tree.
        """
        return self.call(inputs, training=training, mask=mask)

    def add_variable(self, name, shape=None, dtype=jnp.float32, initial_value=None, trainable=True, **kwargs):
        if initial_value is None:
            if shape is None:
                raise ValueError(f"Either initial_value or shape must be provided for variable '{name}'")
            initial_value = jnp.zeros(shape, dtype=dtype)
        var = Variable(initial_value=initial_value, trainable=trainable,
                       name=f"{self.name}_{name}", dtype=dtype, **kwargs)
        self._params[name] = var
        return var

    def add_sub_layer(self, layer_name, layer_obj):
        if not isinstance(layer_obj, Layer):
            raise ValueError("add_sub_layer expects a Layer instance")
        self._sub_layers[layer_name] = layer_obj

    def get_sub_layer(self, layer_name):
        return self._sub_layers.get(layer_name)

    def _get_all_sub_layers(self):
        """
        Gather all sub-layers from explicit registration and instance attributes.
        """
        layers = list(self._sub_layers.values())
        for key, value in self.__dict__.items():
            if isinstance(value, Layer) and value not in layers:
                layers.append(value)
        return layers

    @property
    def variables(self):
        vars_ = list(self._params.values())
        for layer in self._get_all_sub_layers():
            vars_.extend(layer.variables)
        return vars_

    @property
    def trainable_variables(self):
        vars_ = [v for v in self._params.values() if v.trainable]
        for layer in self._get_all_sub_layers():
            vars_.extend(layer.trainable_variables)
        return vars_

    def reset_parameters(self):
        if self.built:
            self.build(self.built_shape)

    def summary(self, print_sub_layers=True):
        lines = []
        lines.append(f"Layer '{self.name}' summary:")
        lines.append(f"  Built: {self.built}, built_shape: {self.built_shape}")
        for key, var in self._params.items():
            lines.append(f"    Param '{key}': shape={var.shape}, dtype={var.dtype}, trainable={var.trainable}")
        if print_sub_layers:
            sub_layers = self._get_all_sub_layers()
            if sub_layers:
                lines.append("  Sub-layers:")
                for sub in sub_layers:
                    lines.append(f"    {sub.name} (built: {sub.built})")
        print("\n".join(lines))

    def get_config(self):
        return {
            "name": self.name,
            "trainable": self.trainable,
            "built": self.built,
            "built_shape": self.built_shape,
            "param_names": list(self._params.keys()),
            "sub_layers": list(self._sub_layers.keys())
        }

    def __repr__(self):
        config = self.get_config()
        return (f"<Layer {config['name']}, built={config['built']}, "
                f"trainable={config['trainable']}, param_count={len(config['param_names'])}>")
