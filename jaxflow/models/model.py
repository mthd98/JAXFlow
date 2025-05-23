# jaxflow/models/model.py
# --------------------------------------------------------------
import math
from typing import Optional, Tuple, Mapping, Any, List, Union

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap, pmap
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from jaxflow.layers.layer import Layer
# --------------------------------------------------------------


class Model(Layer):
    """
    High-performance JAXFlow Model that extends the Layer base class.

    Key Features:
    • Sequential layer composition with explicit `add()` and attribute assignment
    • Lazy building that walks through each layer to infer shapes
    • Pure-functional interface built on Layer's get_params()/set_params()
    • Optax-compatible training with JIT compilation
    • Multi-device support via PMAP
    • Comprehensive training loop with validation and metrics

    Usage:
    ------
    model = Model(name="MyModel")
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Or use attribute assignment (auto-registered)
    model.conv1 = Conv2D(32, kernel_size=3)
    model.pool1 = MaxPooling2D()
    
    model.build(input_shape=(None, 784))
    model.compile(optimizer=optax.adam(1e-3), loss_fn=cross_entropy_loss)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32)
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
        """Auto-register Layer attributes in both sub_layers and sequential layers."""
        super().__setattr__(key, value)
        
        # Auto-add to sequential layers if it's a Layer and not already present
        if (isinstance(value, Layer) and 
            hasattr(self, 'layers') and 
            value not in self.layers and
            not key.startswith('_')):
            self.layers.append(value)

    # ---------------------------------------------------------- #
    # Building and Forward Pass
    # ---------------------------------------------------------- #
    def build(self, input_shape: Tuple[int, ...]):
        """
        Build the model by walking through each layer with dummy data.
        This triggers each layer's build() method and records shapes.
        """
        if self.built:
            return
            
        # Create dummy input for shape inference
        dummy_shape = list(input_shape)
        if dummy_shape[0] in (None, 0):
            dummy_shape[0] = 1
        
        current_shape = tuple(dummy_shape)
        
        # Build each layer sequentially
        for i, layer in enumerate(self.layers):
            if not layer.built:
                layer.build(current_shape)
            
            # Infer output shape by running a forward pass
            dummy_input = jnp.zeros(current_shape, dtype=jnp.float32)
            dummy_output = layer(dummy_input, training=False)
            current_shape = dummy_output.shape
            
        # Mark as built
        self.built = True
        self.built_shape = input_shape
        
        # Pre-compile basic forward functions
        self._compile_forward_functions()

    def call(self, inputs: jnp.ndarray, *, training: bool = False, mask=None) -> jnp.ndarray:
        """Forward pass through all layers sequentially."""
        x = inputs
        current_mask = mask
        
        for layer in self.layers:
            # Handle mask propagation if layer supports it
            if hasattr(layer, 'compute_mask'):
                current_mask = layer.compute_mask(x, current_mask)
                x = layer(x, training=training, mask=current_mask)
            else:
                x = layer(x, training=training)
                
        return x

    def _compile_forward_functions(self):
        """Pre-compile JIT'd forward functions for performance."""
        @jit
        def forward_single(inputs, training: bool):
            return self.call(inputs, training=training)
        
        self._forward_fn = forward_single
        self._batched_forward_fn = vmap(forward_single, in_axes=(0, None))

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
            return self.call(inputs, training=training, mask=mask)
        finally:
            pass
            """# Restore original parameters
            self.set_params(original_params)"""

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
        def eval_step(params, batch_x, batch_y):
            predictions = self.functional_call(batch_x, params, training=False)
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
                val_loss = self.evaluate(*validation_data, batch_size=batch_size, verbose=0, params=params)
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
        params: Optional[Mapping[str, Any]] = None
    ) -> float:
        """Evaluate the model on given data."""
        if params is None:
            params = self.get_params(trainable_only=True)
        
        n_samples = X.shape[0]
        steps = math.ceil(n_samples / batch_size)
        total_loss = 0.0
        
        progress_bar = tqdm(range(steps), desc="Evaluating") if verbose >= 1 else range(steps)
        
        for step in progress_bar:
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_x = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            
            loss_val = self._eval_step_fn(params, batch_x, batch_y)
            total_loss += float(loss_val)
            
            if verbose >= 1 and hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        return total_loss / steps

    def predict(self, X: jnp.ndarray, *, batch_size: Optional[int] = None) -> jnp.ndarray:
        """Generate predictions for input data."""
        if batch_size is None:
            # Process all at once
            return self._batched_forward_fn(X, False)
        
        # Process in batches
        n_samples = X.shape[0]
        steps = math.ceil(n_samples / batch_size)
        predictions = []
        
        for step in range(steps):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = X[start_idx:end_idx]
            batch_pred = self._batched_forward_fn(batch_x, False)
            predictions.append(batch_pred)
        
        return jnp.concatenate(predictions, axis=0)

    def predict_on_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """Generate predictions for a single batch."""
        return self._batched_forward_fn(X, False)

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
    def summary(self, *, show_params: bool = True):
        """Print a summary of the model architecture."""
        print(f"\nModel: {self.name}")
        print("=" * 60)
        
        total_params = 0
        trainable_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_params = len(layer.variables)
            layer_trainable = len(layer.trainable_variables)
            
            print(f"Layer {i:2d}: {layer.name:20s} | params: {layer_params:6d} | trainable: {layer_trainable:6d}")
            
            if show_params and hasattr(layer, 'built_shape') and layer.built_shape:
                print(f"         {'':20s} | shape: {layer.built_shape}")
            
            total_params += layer_params
            trainable_params += layer_trainable
        
        print("=" * 60)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")

    def get_layer(self, name: str) -> Optional[Layer]:
        """Get a layer by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def __len__(self) -> int:
        """Return the number of layers."""
        return len(self.layers)

    def __getitem__(self, index: int) -> Layer:
        """Get layer by index."""
        return self.layers[index]

    def __repr__(self) -> str:
        status = "built" if self.built else "not built"
        compiled_status = "compiled" if self._compiled else "not compiled"
        return f"<Model '{self.name}': {len(self.layers)} layers, {status}, {compiled_status}>"