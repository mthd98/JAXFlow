import jax
import jax.numpy as jnp
import abc
import inspect
from jaxflow.core.variable import Variable

class Layer(abc.ABC):
    """
    Abstract base class for all neural network layers in JAXFlow.

    Handles:
      - Parameter declaration and storage (via `add_variable`).
      - Automatic registration of sub-layers (via attribute assignment and `add_sub_layer`).
      - Lazy building on first call (`__call__` triggers `build`).
      - Mask propagation support (override `compute_mask`).
      - Pure-functional vs. object-oriented forward APIs for JIT compatibility.
      - Utilities for inspecting, resetting, and summarizing parameters and sub-layers.

    Subclasses must implement:
      - `build(input_shape)`: create Variables and sub-layers based on the input shape.
      - `call(inputs, training=False, mask=None)`: define the forward computation.
    """
    def __init__(self, name: str = None, trainable: bool = True):
        # Layer identity and trainability
        self.name = name or self.__class__.__name__
        self.trainable = trainable
        # Internal storage for this layer's Variables and explicit sub-layers
        self._params: dict[str, Variable] = {}
        self._sub_layers: dict[str, Layer] = {}
        # Build state
        self.built: bool = False
        self.built_shape = None

    def __setattr__(self, key, value):
        """
        Auto-register any Layer assigned as an attribute under `_sub_layers`.
        """
        super().__setattr__(key, value)
        if isinstance(value, Layer) and '_sub_layers' in self.__dict__:
            # Avoid duplicates; use attribute name as key
            if key not in self._sub_layers:
                self._sub_layers[key] = value

    @abc.abstractmethod
    def build(self, input_shape):
        """
        Construct Variables and sub-layers based on the given `input_shape`.
        Must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def call(self, inputs, training: bool = False, mask=None):
        """
        Forward computation logic for this layer.
        Must be implemented by subclasses.
        """
        pass

    def __call__(self, inputs, training: bool = False, mask=None):
        """
        Entry point for object-oriented forward pass. Automatically builds on first use.
        """
        if not self.built:
            shape = self._infer_input_shape(inputs)
            self.build(shape)
            self.built = True
            self.built_shape = shape
        # Dispatch to subclass's call
        sig = inspect.signature(self.call)
        if 'mask' in sig.parameters:
            mask_out = self.compute_mask(inputs, mask)
            return self.call(inputs, training=training, mask=mask_out)
        return self.call(inputs, training=training)

    @staticmethod
    def _infer_input_shape(inputs):
        """
        Infer shape tuple(s) from input array(s).
        """
        if isinstance(inputs, (list, tuple)):
            return [x.shape for x in inputs]
        return inputs.shape

    def compute_mask(self, inputs, mask):
        """
        Default mask propagation: return input mask unchanged.
        Override if the layer changes shape or mask semantics.
        """
        return mask

    def functional_call(self, inputs, params: dict, training: bool = False, mask=None):
        """
        Pure-functional forward pass:
          1. Bind raw arrays from `params` into each Variable.
          2. Recursively bind sub-layer parameters.
          3. Invoke the stateful `call` logic.
          4. Restore original Variable values.
        """
        # Backup original values
        original = {n: v.value for n, v in self._params.items()}
        # Bind new values
        for n, v in self._params.items():
            v.value = params[n]
        # Bind sub-layers
        for name, sub in self._sub_layers.items():
            sub.functional_call_placeholder(params.get(name, {}))
        # Execute forward
        sig = inspect.signature(self.call)
        if 'mask' in sig.parameters:
            out = self.call(inputs, training=training, mask=mask)
        else:
            out = self.call(inputs, training=training)
        # Restore originals
        for n, v in self._params.items():
            v.value = original[n]
        return out

    def functional_call_placeholder(self, params: dict):
        """
        Helper to bind raw array values into Variables and sub-layers without calling.
        """
        for n, v in self._params.items():
            v.value = params[n]
        for name, sub in self._sub_layers.items():
            sub.functional_call_placeholder(params.get(name, {}))

    def add_variable(self, name: str, shape=None, dtype=jnp.float32,
                     initial_value=None, trainable: bool = True, **kwargs):
        """
        Declare and store a new Variable in this layer.
        Returns the created Variable.
        """
        if initial_value is None:
            if shape is None:
                raise ValueError(
                    f"Provide `shape` or `initial_value` for variable '{name}'"
                )
            initial_value = jnp.zeros(shape, dtype=dtype)
        var = Variable(
            initial_value=initial_value,
            trainable=trainable,
            name=f"{self.name}_{name}",
            dtype=dtype,
            **kwargs
        )
        self._params[name] = var
        return var

    def add_sub_layer(self, layer_name: str, layer_obj: 'Layer'):
        """
        Explicitly register a child layer under the given `layer_name`.
        """
        if not isinstance(layer_obj, Layer):
            raise ValueError("add_sub_layer expects a Layer instance")
        self._sub_layers[layer_name] = layer_obj

    def _get_all_sub_layers(self):
        """
        Gather both explicit and attribute-assigned sub-layers.
        """
        subs = list(self._sub_layers.values())
        for key, val in self.__dict__.items():
            if isinstance(val, Layer) and val not in subs:
                subs.append(val)
        return subs

    @property
    def variables(self):
        """
        List all Variables in this layer and its sub-layers.
        """
        vars_list = list(self._params.values())
        for sub in self._get_all_sub_layers():
            vars_list.extend(sub.variables)
        return vars_list

    

    @property
    def trainable_variables(self):
        """
        List only trainable Variables in this layer and its sub-layers.
        """
        vars_list = [v for v in self._params.values() if v.trainable]
        for sub in self._get_all_sub_layers():
            vars_list.extend(sub.trainable_variables)
        return vars_list

    def reset_parameters(self):
        """
        Re-run `build` with the same shape to reset Variables.
        """
        if self.built:
            self.build(self.built_shape)

    def summary(self, print_sub_layers: bool = True):
        """
        Print a human-readable summary of this layer’s Variables and sub-layers.
        """
        lines = [f"Layer '{self.name}' summary:",
                 f"  Built: {self.built}, built_shape: {self.built_shape}"]
        for n, v in self._params.items():
            lines.append(
                f"    Param '{n}': shape={v.shape}, dtype={v.dtype}, trainable={v.trainable}"
            )
        if print_sub_layers:
            subs = self._get_all_sub_layers()
            if subs:
                lines.append("  Sub-layers:")
                for sub in subs:
                    lines.append(f"    {sub.name} (built: {sub.built})")
        print("\n".join(lines))

    def get_config(self) -> dict:
        """
        Return a serialization-friendly config dict for this layer.
        """
        return {
            "name": self.name,
            "trainable": self.trainable,
            "built": self.built,
            "built_shape": self.built_shape,
            "param_names": list(self._params.keys()),
            "sub_layers": list(self._sub_layers.keys()),
        }

    def __repr__(self) -> str:
        """
        Return a brief string representation of the layer.
        """
        cfg = self.get_config()
        return (
            f"<Layer {cfg['name']}, built={cfg['built']}, "
            f"trainable={cfg['trainable']}, params={len(cfg['param_names'])}>"
        )
