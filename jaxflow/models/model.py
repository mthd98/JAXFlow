# jaxflow/models/model.py
# --------------------------------------------------------------
import math
from typing import Optional, Tuple, Mapping, Any, List, Union

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap, pmap
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# --------------------------------------------------------------
from jaxflow.layers.layer import Layer

class Model(Layer):
    """
    JAXFlow Model: High-performance, Keras-inspired neural network container.

    The `Model` class organizes and manages a sequence of JAXFlow layers,
    providing both object-oriented and pure-functional APIs for building,
    training, evaluating, and deploying deep learning models. It enables seamless
    integration with JAX and Optax for fast, JIT-compiled, and multi-device training.

    Key Features:
        - Layer management via explicit `add()` or attribute assignment.
        - Lazy shape inference and layer building with automatic variable initialization.
        - Keras-like `compile`, `fit`, `evaluate`, and `predict` methods.
        - JIT-compiled training, evaluation, and inference functions for efficiency.
        - Support for functional-style calls and parameter injection.
        - Multi-device training and inference (pmap/vmap) with optional sharding.
        - Integrated summary, parameter management, and utility helpers.

    Args:
        name (str, optional): Name of the model. Defaults to 'Model'.
        trainable (bool, optional): Whether model parameters are trainable.

    Attributes:
        layers (List[Layer]): List of registered layers in sequential order.
        optimizer: The optimizer instance set by `compile()`.
        loss_fn: The loss function set by `compile()`.
        metrics (List): List of metric functions or objects.
        multi_device (bool): If True, enables multi-device (pmap) prediction.
        built (bool): True if the model and all layers are built.
        built_shape: Shape tuple the model was built with.
        _opt_state: Current optimizer state (internal).
        _compiled (bool): True if model has been compiled.

    Methods:
        add(layer): Add a Layer to the model (sequential API).
        build(input_shape): Build all layers and infer shapes.
        compile(optimizer, loss_fn, metrics=None, multi_device=False): Configure the model for training.
        fit(X, y, epochs, ...): Train the model with optional validation.
        evaluate(X, y, ...): Compute loss on test data.
        predict(X, batch_size=None): Predict outputs for input data.
        predict_on_batch(X): Predict for a single batch.
        predict_pmap(X_sharded): Predict in parallel across multiple devices.
        summary(show_params=True): Print a summary of the model architecture.
        get_layer(name): Retrieve a layer by name.
        get_params(trainable_only=True): Return current model parameters.
        set_params(params): Set model parameters (from dict).
        functional_call(inputs, params, ...): Forward pass with injected parameters.
        __len__(): Number of layers.
        __getitem__(index): Get layer by index.

    Example:
        >>> model = Model(name="MLP")
        >>> model.add(Dense(128, activation=jax.nn.relu))
        >>> model.add(Dense(10, activation=jax.nn.softmax))
        >>> model.build(input_shape=(None, 784))
        >>> optimizer = MyAdam(learning_rate=1e-3)
        >>> loss_fn = SparseCategoricalCrossentropy()
        >>> model.compile(optimizer=optimizer, loss_fn=loss_fn)
        >>> history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
        >>> val_acc = model.evaluate(X_val, y_val)
        >>> preds = model.predict(X_test)

    Example with Class API:
        >>> class MyModel(Model):
        >>>     def __init__(self):
        >>>         super().__init__(name="MyModel")
        >>>         self.d1 = Dense(128, activation=jax.nn.relu)
        >>>         self.d2 = Dense(10, activation=jax.nn.softmax)
        >>>      def call(self, inputs, training=False):
        >>>         x = self.d1(inputs, training=training)
        >>>         return self.d2(x, training=training)
        >>> model = MyModel()
        >>> model.build(input_shape=(None, 784))
        >>> optimizer = MyAdam(learning_rate=1e-3)
        >>> loss_fn = SparseCategoricalCrossentropy()
        >>> model.compile(optimizer=optimizer, loss_fn=loss_fn)
        >>> history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
        >>> val_acc = model.evaluate(X_val, y_val)
        >>> preds = model.predict(X_test)



    Notes:
        - The model supports Keras-like sequential composition and training, but is JAX-native.
        - Models are compiled and JIT-optimized after `build()` and `compile()`.
        - Multi-device (pmap) support is enabled via `compile(multi_device=True)`.
        - Metric APIs are extensible and accept any callable with signature `fn(y_true, y_pred)`.

    Raises:
        ValueError: If an invalid layer is added or shapes are incompatible.
        RuntimeError: If fit/evaluate/predict is called before compile/build.

    See Also:
        Layer, BaseOptimizer, Loss, Metric

    """

    def __init__(self, name: Optional[str] = None, *, trainable: bool = True):
        super().__init__(name=name or "Model", trainable=trainable)
        
        # Layer management
        self.layers: List[Layer] = []  # Sequential execution order
        
        # Training configuration
        self.optimizer = None
        self.loss_fn = None
        self.metrics: List = []
        self.multi_device: bool = False
        
        # Training state
        self._opt_state = None
        self._compiled = False
        self._is_subclassed = False  # Track if this is a subclassed model
        
        # Compiled functions (created during compile())
        self._train_step_fn = None
        self._eval_step_fn = None
        self._forward_fn = None
        self._batched_forward_fn = None
        self._parallel_forward_fn = None

    # ---------------------------------------------------------- #
    # Layer Management
    # ---------------------------------------------------------- #
    def add(self, layer: Layer) -> 'Model':
        """Add a layer to the sequential execution chain."""
        if not isinstance(layer, Layer):
            raise ValueError(f"Expected Layer instance, got {type(layer)}")
        
        self.layers.append(layer)
        # Also register as sub-layer for parameter management
        layer_name = f"layer_{len(self.layers)-1}"
        self._sub_layers[layer_name] = layer
        return self

    def __setattr__(self, key: str, value: Any):
        """Auto-register Layer attributes for parameter management."""
        super().__setattr__(key, value)
        
        # Auto-register layers assigned as attributes (for subclassing API)
        if (isinstance(value, Layer) and 
            hasattr(self, '_sub_layers') and  # Make sure _sub_layers exists
            not key.startswith('_') and 
            key not in ['layers']):  # Don't register the layers list itself
            
            # Register in sub_layers for parameter management
            self._sub_layers[key] = value
            
            # For subclassed models, don't auto-add to sequential layers
            # The user defines the execution order in their call() method
            if not self._is_subclassed and hasattr(self, 'layers'):
                # Only add to sequential layers if not already present
                if value not in self.layers:
                    self.layers.append(value)

    def _detect_subclassing(self):
        """Detect if this model uses subclassing API."""
        # Check if call method is overridden
        call_method = getattr(self.__class__, 'call', None)
        model_call_method = getattr(Model, 'call', None)
        
        if call_method is not model_call_method:
            self._is_subclassed = True
        
        # Also check if there are any Layer attributes defined
        layer_attrs = [attr for attr, value in self.__dict__.items() 
                      if isinstance(value, Layer) and not attr.startswith('_')]
        
        if layer_attrs and not self.layers:
            self._is_subclassed = True

    # ---------------------------------------------------------- #
    # Building and Forward Pass
    # ---------------------------------------------------------- #
    def build(self, input_shape: Tuple[int, ...]):
        """
        Build the model by walking through layers or using dummy forward pass.
        Handles both sequential and subclassed models.
        """
        if self.built:
            return
        
        # Detect if this is a subclassed model
        self._detect_subclassing()
        
        # Create dummy input for shape inference
        dummy_shape = list(input_shape)
        if dummy_shape[0] in (None, 0):
            dummy_shape[0] = 1
        
        dummy_input = jnp.zeros(tuple(dummy_shape), dtype=jnp.float32)
        
        if self._is_subclassed:
            # For subclassed models, do a forward pass to build all layers
            try:
                _ = self.call(dummy_input, training=False)
            except Exception as e:
                print(f"Warning: Could not build model with dummy forward pass: {e}")
                # Fallback: try to build individual layer attributes
                self._build_layer_attributes(tuple(dummy_shape))
        else:
            # For sequential models, build each layer in order
            current_shape = tuple(dummy_shape)
            
            for i, layer in enumerate(self.layers):
                if not layer.built:
                    layer.build(current_shape)
                
                # Infer output shape by running a forward pass
                dummy_layer_input = jnp.zeros(current_shape, dtype=jnp.float32)
                dummy_output = layer(dummy_layer_input, training=False)
                current_shape = dummy_output.shape
        
        # Mark as built
        self.built = True
        self.built_shape = input_shape
        
        # Pre-compile basic forward functions
        self._compile_forward_functions()

    def _build_layer_attributes(self, input_shape: Tuple[int, ...]):
        """Build individual layer attributes for subclassed models."""
        for attr_name, layer in self._sub_layers.items():
            if isinstance(layer, Layer) and not layer.built:
                try:
                    layer.build(input_shape)
                except Exception as e:
                    print(f"Warning: Could not build layer {attr_name}: {e}")

    def call(self, inputs: jnp.ndarray, *, training: bool = False, mask=None, **kwargs) -> jnp.ndarray:
        """
        Forward pass through the model.
        This is the default implementation for sequential models.
        Subclassed models should override this method.
        """
        if self._is_subclassed:
            # This should be overridden in subclassed models
            raise NotImplementedError(
                "Subclassed models must implement their own call() method. "
                "Define how data flows through your layers."
            )
        
        # Default sequential implementation
        x = inputs
        current_mask = mask
        
        for layer in self.layers:
            # Handle mask propagation if layer supports it
            if hasattr(layer, 'compute_mask'):
                current_mask = layer.compute_mask(x, current_mask)
                x = layer(x, training=training, mask=current_mask)
            else:
                # For layers that don't support masks, pass only training parameter
                x = layer(x, training=training)
                
        return x

    def _compile_forward_functions(self):
        """Pre-compile JIT'd forward functions for performance."""
        @jit
        def forward_single(inputs, training: bool, mask=None):
            return self.call(inputs, training=training, mask=mask)
        
        def forward_batch(inputs_batch, training: bool, mask_batch=None):
            # Handle batched masks if provided
            if mask_batch is not None:
                return vmap(forward_single, in_axes=(0, None, 0))(inputs_batch, training, mask_batch)
            else:
                return vmap(forward_single, in_axes=(0, None, None))(inputs_batch, training, None)
        
        self._forward_fn = forward_single
        self._batched_forward_fn = forward_batch

    # ---------------------------------------------------------- #
    # Pure Functional Interface
    # ---------------------------------------------------------- #
    def functional_call(
        self, 
        inputs: jnp.ndarray, 
        params: Mapping[str, Any], 
        *, 
        training: bool = False,
        mask=None
    ) -> jnp.ndarray:
        """
        Pure functional forward pass with parameter injection.
        This temporarily sets parameters and runs the forward pass.
        """
        # Store current parameters
        original_params = self.get_params(trainable_only=False)
        
        try:
            # Inject new parameters
            self.set_params(params)
            # Run forward pass
            if mask is  None:
                return self.call(inputs, training=training)
            else:
                return self.call(inputs, training=training,mask=mask)
        finally:
            # Restore original parameters (commented out as in original)
            pass

    # ---------------------------------------------------------- #
    # Training Configuration
    # ---------------------------------------------------------- #
    def compile(
        self,
        optimizer,
        loss_fn,
        *,
        metrics: Optional[List] = None,
        multi_device: bool = False
    ):
        """Configure the model for training."""
        if not self.built:
            raise RuntimeError("Model must be built before compiling. Call build() first.")
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.multi_device = multi_device
        
        # Initialize optimizer state
        params = self.get_params(trainable_only=True)
        self._opt_state = self.optimizer.init(params)
        
        # Compile training functions
        self._compile_training_functions()
        
        # Setup multi-device support
        if multi_device:
            self._parallel_forward_fn = pmap(self._batched_forward_fn, in_axes=(0, None))
        
        self._compiled = True

    def _compile_training_functions(self):
        """Compile JIT'd training and evaluation step functions."""
        
        @jit
        def train_step(params, opt_state, batch_x, batch_y):
            def loss_fn(p):
                predictions = self.functional_call(batch_x, p, training=True)
                return self.loss_fn(batch_y, predictions)
            
            loss_val, grads = value_and_grad(loss_fn)(params)
            new_params, new_opt_state = self.optimizer.update(grads, opt_state, params)
            
            return new_params, new_opt_state, loss_val
        
        @jit
        def eval_step(batch_x, batch_y, p):
            predictions = self.functional_call(batch_x, p, training=False)
            return self.loss_fn(batch_y, predictions)
        
        self._train_step_fn = train_step
        self._eval_step_fn = eval_step

    # ---------------------------------------------------------- #
    # Training Loop
    # ---------------------------------------------------------- #
    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        epochs: int,
        batch_size: int = 32,
        validation_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        validation_split: Optional[float] = None,
        verbose: int = 1,
        shuffle: bool = True
    ) -> dict:
        """Train the model."""
        if not self._compiled:
            raise RuntimeError("Model must be compiled before training. Call compile() first.")
        
        # Handle validation split
        if validation_split is not None:
            if validation_data is not None:
                raise ValueError("Cannot specify both validation_data and validation_split")
            X, X_val, y, y_val = train_test_split(X, y, test_size=validation_split, shuffle=shuffle)
            validation_data = (X_val, y_val)
        
        n_samples = X.shape[0]
        steps_per_epoch = math.ceil(n_samples / batch_size)
        
        # Training state
        params = self.get_params(trainable_only=True)
        opt_state = self._opt_state
        
        # History tracking
        history = {"loss": [], "epoch": []}
        if validation_data is not None:
            history["val_loss"] = []
        
        # Training loop
        for epoch in range(1, epochs + 1):
            if verbose >= 1:
                print(f"\nEpoch {epoch}/{epochs}")
            
            # Shuffle data if requested
            if shuffle:
                perm = jax.random.permutation(jax.random.PRNGKey(epoch), n_samples)
                X_epoch, y_epoch = X[perm], y[perm]
            else:
                X_epoch, y_epoch = X, y
            
            # Training batches
            epoch_loss = 0.0
            progress_bar = tqdm(range(steps_per_epoch), desc="Training") if verbose >= 1 else range(steps_per_epoch)
            
            for step in progress_bar:
                start_idx = step * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_x = X_epoch[start_idx:end_idx]
                batch_y = y_epoch[start_idx:end_idx]
                
                # Training step
                params, opt_state, loss_val = self._train_step_fn(params, opt_state, batch_x, batch_y)
                epoch_loss += float(loss_val)
                
                if verbose >= 1 and hasattr(progress_bar, 'set_postfix'):
                    progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})
            
            # Record training metrics
            avg_loss = epoch_loss / steps_per_epoch
            history["loss"].append(avg_loss)
            history["epoch"].append(epoch)
            
            # Update model parameters and optimizer state
            self.set_params(params)
            self._opt_state = opt_state

            # Validation
            if validation_data is not None:
                val_loss = self.evaluate(*validation_data, batch_size=batch_size, verbose=0)
                history["val_loss"].append(val_loss)
                
                if verbose >= 1:
                    print(f"loss: {avg_loss:.4f} - val_loss: {val_loss:.4f}")
            elif verbose >= 1:
                print(f"loss: {avg_loss:.4f}")
        
        return history

    # ---------------------------------------------------------- #
    # Evaluation and Prediction
    # ---------------------------------------------------------- #
    def evaluate(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> float:
        """Evaluate the model on given data."""
        n_samples = X.shape[0]
        steps = math.ceil(n_samples / batch_size)
        total_loss = 0.0
        params = self.get_params(trainable_only=True)
        
        progress_bar = tqdm(range(steps), desc="Evaluating") if verbose >= 1 else range(steps)
        
        for step in progress_bar:
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_x = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            
            loss_val = self._eval_step_fn(batch_x, batch_y, params)
            total_loss += float(loss_val)
            
            if verbose >= 1 and hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})

        self.set_params(params)
        return total_loss / steps

    def predict(self, X: jnp.ndarray, *, batch_size: Optional[int] = None, mask=None) -> jnp.ndarray:
        """Generate predictions for input data."""
        if batch_size is None:
            # Process all at once
            if mask is None:
                return self._batched_forward_fn(X, False)
            return self._batched_forward_fn(X, False, mask)
        
        # Process in batches
        n_samples = X.shape[0]
        steps = math.ceil(n_samples / batch_size)
        predictions = []
        
        for step in range(steps):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = X[start_idx:end_idx]
            
            # Handle batched masks
            batch_mask = None
            if mask is not None:
                batch_mask = mask[start_idx:end_idx]
                
            batch_pred = self._batched_forward_fn(batch_x, False, batch_mask)
            predictions.append(batch_pred)
        
        return jnp.concatenate(predictions, axis=0)

    def predict_on_batch(self, X: jnp.ndarray, *, mask=None) -> jnp.ndarray:
        """Generate predictions for a single batch."""
        if mask is None:
            return self._batched_forward_fn(X, False)
        return self._batched_forward_fn(X, False, mask)

    # ---------------------------------------------------------- #
    # Multi-device Support
    # ---------------------------------------------------------- #
    def predict_pmap(self, X_sharded: jnp.ndarray) -> jnp.ndarray:
        """Parallel prediction across multiple devices."""
        if not self.multi_device:
            raise RuntimeError("Model must be compiled with multi_device=True for PMAP operations")
        
        return self._parallel_forward_fn(X_sharded, False)

    # ---------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------- #
    def summary(self, *, line_length: int = 65):
        """
        Print a summary of the model architecture in Keras-style format.
        
        Args:
            line_length: Total length of printed lines (default: 65)
        """
        def format_number(num):
            """Format parameter count with commas."""
            return f"{num:,}"
        
        def get_shape_string(shape):
            """Convert shape tuple to formatted string."""
            if shape is None:
                return "(Unknown)"
            
            if isinstance(shape, (list, tuple)):
                # Replace None with ? for batch dimension, keep others as-is
                formatted_shape = tuple('?' if dim is None else dim for dim in shape)
                return str(formatted_shape)
            
            return str(shape)
        
        def calculate_layer_params(layer):
            """Calculate parameters for a layer accurately."""
            param_count = 0
            
            try:
                if hasattr(layer, 'get_params'):
                    params = layer.get_params(trainable_only=False)
                    for param_array in params.values():
                        if hasattr(param_array, 'size'):
                            param_count += int(param_array.size)
                        elif hasattr(param_array, 'shape'):
                            import numpy as np
                            param_count += int(np.prod(param_array.shape))
                elif hasattr(layer, 'variables'):
                    for var in layer.variables:
                        if hasattr(var, 'size'):
                            param_count += int(var.size)
                        elif hasattr(var, 'shape'):
                            import numpy as np
                            param_count += int(np.prod(var.shape))
            except Exception:
                param_count = 0
            
            return param_count
        
        def get_layer_output_shape(layer, layer_index=None):
            """Get output shape for a layer."""
            # Try built_shape first
            if hasattr(layer, 'built_shape') and layer.built_shape:
                return get_shape_string(layer.built_shape)
            
            # Try output_shape attribute
            if hasattr(layer, 'output_shape'):
                return get_shape_string(layer.output_shape)
            
            # For sequential models, try to infer from input
            if not self._is_subclassed and layer_index is not None:
                if layer_index == 0 and hasattr(self, 'built_shape') and self.built_shape:
                    # First layer: try to compute output from model input shape
                    if hasattr(layer, 'compute_output_shape'):
                        try:
                            output_shape = layer.compute_output_shape(self.built_shape)
                            return get_shape_string(output_shape)
                        except Exception:
                            pass
            
            return "(Unknown)"
        
        # Calculate column widths
        name_width = int(line_length * 0.45)
        shape_width = int(line_length * 0.35) 
        param_width = line_length - name_width - shape_width - 4  # 4 for separators and padding
        
        # Model header
        print(f'\nModel: "{self.name}"')
        print("_" * line_length)
        
        # Column headers
        header = f"{'Layer (type)':<{name_width}} {'Output Shape':<{shape_width}} {'Param #':>{param_width}}"
        print(header)
        print("=" * line_length)
        
        # Initialize counters
        total_params = 0
        trainable_params = 0
        
        # Process layers based on model type
        if self._is_subclassed:
            # Subclassed model: show layer attributes
            for attr_name, layer in self._sub_layers.items():
                if isinstance(layer, Layer):
                    # Format layer name with type
                    layer_display = f"{attr_name} ({layer.__class__.__name__})"
                    if len(layer_display) > name_width - 1:
                        layer_display = layer_display[:name_width-4] + "..."
                    
                    # Get output shape
                    output_shape = get_layer_output_shape(layer)
                    if len(output_shape) > shape_width - 1:
                        output_shape = output_shape[:shape_width-4] + "..."
                    
                    # Calculate parameters
                    param_count = calculate_layer_params(layer)
                    total_params += param_count
                    
                    # Check if trainable
                    if hasattr(layer, 'trainable') and layer.trainable:
                        trainable_params += param_count
                    
                    # Print row
                    param_str = format_number(param_count)
                    row = f"{layer_display:<{name_width}} {output_shape:<{shape_width}} {param_str:>{param_width}}"
                    print(row)
        
        else:
            # Sequential model: show layers in order
            for i, layer in enumerate(self.layers):
                # Format layer name with type
                layer_display = f"{layer.name} ({layer.__class__.__name__})"
                if len(layer_display) > name_width - 1:
                    layer_display = layer_display[:name_width-4] + "..."
                
                # Get output shape
                output_shape = get_layer_output_shape(layer, i)
                if len(output_shape) > shape_width - 1:
                    output_shape = output_shape[:shape_width-4] + "..."
                
                # Calculate parameters
                param_count = calculate_layer_params(layer)
                total_params += param_count
                
                # Check if trainable
                if hasattr(layer, 'trainable') and layer.trainable:
                    trainable_params += param_count
                
                # Print row
                param_str = format_number(param_count)
                row = f"{layer_display:<{name_width}} {output_shape:<{shape_width}} {param_str:>{param_width}}"
                print(row)
        
        # Footer with totals
        print("=" * line_length)
        print(f"Total params: {format_number(total_params)}")
        print(f"Trainable params: {format_number(trainable_params)}")  
        print(f"Non-trainable params: {format_number(total_params - trainable_params)}")
        
        # Additional model information
        model_type = "Subclassed" if self._is_subclassed else "Sequential"
        build_status = "built" if self.built else "not built"
        compile_status = "compiled" if self._compiled else "not compiled"
        
        print("_" * line_length)
        print(f"Model type: {model_type} | Status: {build_status}, {compile_status}")
        print()

    def get_layer(self, name: str) -> Optional[Layer]:
        """Get a layer by name."""
        # Check sequential layers
        for layer in self.layers:
            if layer.name == name:
                return layer
        
        # Check layer attributes (for subclassed models)
        for attr_name, layer in self._sub_layers.items():
            if isinstance(layer, Layer) and (layer.name == name or attr_name == name):
                return layer
                
        return None

    def __len__(self) -> int:
        """Return the number of layers."""
        if self._is_subclassed:
            return len([layer for layer in self._sub_layers.values() if isinstance(layer, Layer)])
        return len(self.layers)

    def __getitem__(self, index: int) -> Layer:
        """Get layer by index."""
        if self._is_subclassed:
            layer_list = [layer for layer in self._sub_layers.values() if isinstance(layer, Layer)]
            return layer_list[index]
        return self.layers[index]

    def __repr__(self) -> str:
        model_type = "subclassed" if self._is_subclassed else "sequential"
        status = "built" if self.built else "not built"
        compiled_status = "compiled" if self._compiled else "not compiled"
        return f"<Model '{self.name}' ({model_type}): {len(self)} layers, {status}, {compiled_status}>"