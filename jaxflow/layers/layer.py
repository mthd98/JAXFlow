"""
jaxflow.layers.layer
~~~~~~~~~~~~~~~~~~~~

Base `Layer` class for **JAXFlow** – now powered by a shared
`AutoNameMixin` so *every* object (Layer, Model, Variable, Optimizer…)
follows the same collision-free auto-naming scheme.

Key capabilities
----------------
• Variable declaration (`add_variable`) with automatic name-scoping.
• Transparent sub-layer registration via normal attribute assignment.
• Lazy `build()` called the first time the layer is invoked.
• Object-oriented **and** pure-functional forward APIs.
• Mask propagation hook (`compute_mask`).
• Summaries, parameter resets, config export.
• NEW ⚡ Global auto-naming through `AutoNameMixin`.
"""

from __future__ import annotations

import abc
import inspect
from typing import Dict, Any, Mapping, List, Optional
import jax
import jax.numpy as jnp

from jaxflow.core.variable import Variable
from collections import defaultdict
from threading import Lock
from typing import Optional, Dict
from jaxflow.core.auto_name import AutoNameMixin



class Layer(AutoNameMixin, abc.ABC):
    """
    Abstract base class for all neural network layers in JAXFlow.

    `Layer` implements the foundational API for building custom layers, 
    managing parameters, automatic name scoping, sub-layer registration, 
    and pure-functional forward passes. All JAXFlow layers should inherit 
    from this class to enable composability and maintain consistency with 
    the JAXFlow architecture.

    Key Features:
        - Variable declaration with automatic name scoping.
        - Transparent sub-layer registration via attribute assignment.
        - Lazy `build()` invocation upon first call.
        - Both object-oriented and pure-functional forward APIs.
        - Mask propagation support via `compute_mask`.
        - Utility functions for parameter management and summaries.
        - Global auto-naming via `AutoNameMixin`.

    Args:
        name (str, optional): Name for the layer. If None, a unique name is auto-generated.
        trainable (bool, optional): Whether the layer's variables are trainable. Defaults to True.

    Inputs:
        inputs (Array, PyTree, or Sequence): Input data for the layer. The exact format is determined
            by the subclass implementation.

    Input shape:
        Arbitrary; subclasses define required shape. Commonly, input shape is `(batch_size, features)` 
        for dense layers, or `(batch_size, ..., channels)` for convolutional layers.

    Output shape:
        Arbitrary; subclasses define the output shape. Usually determined by the layer's transformation.

    Attributes:
        name (str): The name of the layer, unique within the model.
        trainable (bool): Whether variables in this layer are trainable.
        variables (list of Variable): All variables in this layer and its sub-layers.
        trainable_variables (list of Variable): All trainable variables in this layer and sub-layers.
        built (bool): Whether the layer has been built.
        built_shape (Any): The shape(s) with which the layer was built.
        _params (dict): Mapping of variable names to Variable objects.
        _sub_layers (dict): Mapping of sub-layer names to Layer objects.

    Example:
        ```python
        class MyDense(Layer):
            def __init__(self, units, name=None, trainable=True):
                super().__init__(name=name, trainable=trainable)
                self.units = units

            def build(self, input_shape):
                self.add_variable('kernel', shape=(input_shape[-1], self.units))
                self.add_variable('bias', shape=(uself.nits,))
            def call(self, inputs, training=False, mask=None):
                kernel = self._params['kernel'].value
                bias = self._params['bias'].value
                return jnp.dot(inputs, kernel) + bias

        layer = MyDense(name='dense_1')
        output = layer(jnp.ones((32, 128)))
        ```

    Raises:
        NotImplementedError: If `build` or `call` are not implemented by subclass.
        ValueError: If required arguments (e.g., for variable initialization) are missing.

    Note:
        This class is intended to be subclassed. Subclasses must implement `build` and `call`.
    """


    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, name: str | None = None, *, trainable: bool = True):
        # Consistent, thread-safe auto-naming via mix-in
        self.name: str = self.auto_name(name)
        self.trainable: bool = trainable

        # Internal bookkeeping
        self._params: Dict[str, Variable] = {}     # local Variables
        self._sub_layers: Dict[str, "Layer"] = {}  # direct children

        self.built: bool = False
        self.built_shape: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Automatic sub-layer registration
    # ------------------------------------------------------------------ #
    def __setattr__(self, key: str, value: Any):
        super().__setattr__(key, value)
        # After __init__, ensure we auto-register new Layer attrs
        if isinstance(value, Layer) and "_sub_layers" in self.__dict__:
            self._sub_layers.setdefault(key, value)

    # ------------------------------------------------------------------ #
    # Abstract hooks to implement in subclasses
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def build(self, input_shape):
        """Allocate Variables & sub-layers from `input_shape`."""
        ...

    @abc.abstractmethod
    def call(self, inputs, *, training: bool = False, mask=None):
        """Forward computation logic (OO API)."""
        ...

    # ------------------------------------------------------------------ #
    # Public forward entry-point
    # ------------------------------------------------------------------ #
    def __call__(self, inputs, *, training: bool = False, mask=None):
        # Lazy build
        if not self.built:
            shape = self._infer_input_shape(inputs)
            self.build(shape)
            self.built = True
            self.built_shape = shape

        # Dispatch to subclass `call`
        sig = inspect.signature(self.call)
        if "mask" in sig.parameters:
            mask_out = self.compute_mask(inputs, mask)
            return self.call(inputs, training=training, mask=mask_out)
        return self.call(inputs, training=training)

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_input_shape(inputs):
        """Return py-tree of shapes matching the inputs."""
        if isinstance(inputs, (list, tuple)):
            return [x.shape for x in inputs]
        return inputs.shape

    def compute_mask(self, inputs, mask):
        """Default behaviour: propagate mask unchanged."""
        return mask

    # ------------------------------------------------------------------ #
    # Pure-functional forward helpers
    # ------------------------------------------------------------------ #
    def functional_call(
        self,
        inputs,
        params: Mapping[str, Any],
        *,
        training: bool = False,
        mask=None,
    ):
        """Pure-functional forward – swaps variable values temporarily."""
        original = self.get_params(trainable_only=False)
        try:
            self.set_params(params)
            sig = inspect.signature(self.call)
            if "mask" in sig.parameters:
                return self.call(inputs, training=training, mask=mask)
            return self.call(inputs, training=training)
        finally:
            self.set_params(original)

    def functional_call_placeholder(self, params: Mapping[str, Any]):
        """Bind params without running forward – for recursive builders."""
        self.set_params(params)

    # ------------------------------------------------------------------ #
    # Variable & sub-layer helpers
    # ------------------------------------------------------------------ #
    def add_variable(
        self,
        name: str,
        *,
        shape=None,
        dtype=jnp.float32,
        initial_value=None,
        trainable: bool = True,
        **kwargs,
    ) -> Variable:
        """Declare & register a new Variable."""
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
            **kwargs,
        )
        self._params[name] = var
        return var

    def add_sub_layer(self, layer_name: str, layer_obj: "Layer"):
        """Explicit registration when not using attribute assignment."""
        if not isinstance(layer_obj, Layer):
            raise ValueError("add_sub_layer expects a Layer instance")
        self._sub_layers[layer_name] = layer_obj

    def _get_all_sub_layers(self) -> List["Layer"]:
        subs = list(self._sub_layers.values())
        for val in self.__dict__.values():
            if isinstance(val, Layer) and val not in subs:
                subs.append(val)
        return subs

    # ------------------------------------------------------------------ #
    # Variable collections
    # ------------------------------------------------------------------ #
    @property
    def variables(self) -> List[Variable]:
        vars_ = list(self._params.values())
        for sub in self._get_all_sub_layers():
            vars_.extend(sub.variables)
        return vars_

    @property
    def trainable_variables(self) -> List[Variable]:
        vars_ = [v for v in self._params.values() if v.trainable]
        for sub in self._get_all_sub_layers():
            vars_.extend(sub.trainable_variables)
        return vars_

    # ------------------------------------------------------------------ #
    # Parameter (de)serialisation
    # ------------------------------------------------------------------ #
    def _collect_params(self, trainable_only: bool = True) -> Dict[str, Any]:
        tree: Dict[str, Any] = {
            n: var.value for n, var in self._params.items()
            if (not trainable_only or var.trainable)
        }
        for sub in self._sub_layers.values():
            sub_tree = sub._collect_params(trainable_only)
            if sub_tree:
                tree[sub.name] = sub_tree
        return tree

    def get_params(
    self,
    trainable_only: bool = True,
    *,
    concrete: bool = False,
    ) -> Dict[str, Any]:
        """
        Return a PyTree of parameters.

        Args
        ----
        trainable_only : if ``True`` return only trainable vars.
        concrete       : if ``True`` run `jax.device_get` on every leaf
                        so the result is guaranteed to be a DeviceArray
                        (never a Tracer), even when called inside a
                        ``jit`` / ``grad`` context.
        """
        tree = self._collect_params(trainable_only)
        if concrete:
            tree = jax.tree.map(jax.device_get, tree)
        return tree

    def _apply_params(self, tree: Mapping[str, Any]):
        for name, var in self._params.items():
            if name in tree:
                var.assign(tree[name])
        for sub in self._sub_layers.values():
            if sub.name in tree:
                sub._apply_params(tree[sub.name])

    def set_params(self, tree: Mapping[str, Any]):
        self._apply_params(tree)

    # ------------------------------------------------------------------ #
    # Maintenance helpers
    # ------------------------------------------------------------------ #
    def reset_parameters(self):
        if self.built:
            self.build(self.built_shape)

    def summary(self, *, print_sub_layers: bool = True):
        lines = [
            f"Layer '{self.name}' — built: {self.built}, shape: {self.built_shape}"
        ]
        for n, v in self._params.items():
            lines.append(
                f"  • Param '{n}': shape={v.shape}, dtype={v.dtype}, trainable={v.trainable}"
            )
        if print_sub_layers and self._sub_layers:
            lines.append("  • Sub-layers:")
            for sub in self._sub_layers.values():
                lines.append(f"    – {sub.name} (built: {sub.built})")
        print("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Serialisation & repr
    # ------------------------------------------------------------------ #
    def get_config(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            trainable=self.trainable,
            built=self.built,
            built_shape=self.built_shape,
            param_names=list(self._params.keys()),
            sub_layers=list(self._sub_layers.keys()),
        )

    def __repr__(self):
        cfg = self.get_config()
        return (
            f"<Layer {cfg['name']} | built={cfg['built']} "
            f"| trainable={cfg['trainable']} | params={len(cfg['param_names'])}>"
        )
