# jaxflow/layers/gru.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from jaxflow.layers.layer import Layer
from jaxflow.initializers import GlorotUniform, Orthogonal, Zeros

class GRU(Layer):
    """
    Gated Recurrent Unit (GRU) layer for JAXFlow (Cho et al., 2014).

    Implements the classic GRU update:
        h_t = (1 − z_t) * h_{t−1} + z_t * h̃_t

    Can return either the last output for each batch or the full sequence, with
    options to return the final hidden state. Follows Keras/PyTorch semantics
    for parameter names and behavior.

    Args:
        units (int): Dimensionality of the output space (number of hidden units).
        activation (callable, optional): Activation function for the candidate hidden state.
            Defaults to tanh.
        recurrent_activation (callable, optional): Activation function for the update/reset gates.
            Defaults to sigmoid.
        use_bias (bool, optional): Whether the layer uses bias vectors. Defaults to True.
        return_sequences (bool, optional): If True, returns the full sequence of outputs (batch, time, units).
            If False (default), returns only the last output (batch, units).
        return_state (bool, optional): If True, returns a tuple (output, last_hidden_state).
        device (str, optional): Device for parameter placement ("auto", "cpu", "gpu", "tpu"). Defaults to "auto".
        shard_devices (list or str, optional): Devices for parameter sharding. See Variable docs.
        dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
        trainable (bool, optional): Whether the parameters are trainable. Defaults to True.
        kernel_initializer (callable or Initializer, optional): Initializer for input kernel weights.
            Defaults to GlorotUniform.
        recurrent_initializer (callable or Initializer, optional): Initializer for recurrent kernel weights.
            Defaults to Orthogonal.
        bias_initializer (callable or Initializer, optional): Initializer for bias vectors.
            Defaults to Zeros.
        seed (int, optional): Random seed for parameter initialization.
        name (str, optional): Layer name. If None, a unique name is generated.

    Inputs:
        inputs (jnp.ndarray): 3D tensor of shape (batch, time, features).
        initial_state (jnp.ndarray, optional): Initial hidden state. Shape: (batch, units).
        mask (jnp.ndarray, optional): Boolean tensor with shape (batch, time), for masking input steps.

    Input shape:
        (batch_size, time_steps, features)

    Output shape:
        (batch_size, units) if return_sequences=False (default)
        (batch_size, time_steps, units) if return_sequences=True

        If return_state=True, returns a tuple: (output, last_hidden_state)

    Attributes:
        units (int): Dimensionality of the output space.
        activation (callable): Activation for candidate state.
        recurrent_activation (callable): Activation for update/reset gates.
        use_bias (bool): Whether bias is used.
        return_sequences (bool): Whether to return the full output sequence.
        return_state (bool): Whether to return the last hidden state.
        kernel (Variable): Input kernel weights.
        recurrent_kernel (Variable): Recurrent kernel weights.
        bias (Variable): Bias vector (if use_bias is True).
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from jaxflow.layers.gru import GRU

        # Example input: batch of 8, sequence length 10, feature size 32
        x = jnp.ones((8, 10, 32))
        gru = GRU(units=16, return_sequences=True)
        y = gru(x)  # (8, 10, 16)
        ```

    Raises:
        ValueError: If input shape does not match (batch, time, features).

    Note:
        - No explicit stateful mode; for stateful inference, manually manage the initial_state.
        - Input kernel and recurrent kernel are concatenated for efficiency ([z, r, h̃]).
        - Compatible with JAX JIT/vmap/pmap for efficient batching and functional API use.
    """


    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        units: int,
        *,
        name: str | None = None,
        activation=jnp.tanh,
        recurrent_activation=jax.nn.sigmoid,
        use_bias: bool = True,
        return_sequences: bool = False,
        return_state: bool = False,
        device: str = "auto",
        shard_devices=None,
        dtype=jnp.float32,
        trainable: bool = True,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        seed: int | None = None,
    ):
        super().__init__(name=name, trainable=trainable)

        # Public config
        self.units = int(units)
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = bool(use_bias)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)

        # Initialisers
        kernel_initializer = kernel_initializer or GlorotUniform
        recurrent_initializer = recurrent_initializer or Orthogonal
        bias_initializer = bias_initializer or Zeros

        self.kernel_init = kernel_initializer(seed=seed, dtype=dtype) if callable(
            kernel_initializer
        ) else kernel_initializer
        self.recurrent_init = (
            recurrent_initializer(seed=seed, dtype=dtype)
            if callable(recurrent_initializer)
            else recurrent_initializer
        )
        self.bias_init = bias_initializer(seed=seed, dtype=dtype) if callable(
            bias_initializer
        ) else bias_initializer

        # Misc
        self.device = device
        self.shard_devices = shard_devices
        self.dtype = dtype
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("GRU expects inputs shaped (batch, time, features).")
        _, _, in_features = input_shape

        # Kernels concatenated as [z, r, h̃]
        self.kernel = self.add_variable(
            "kernel",
            initial_value=self.kernel_init(shape=(in_features, 3 * self.units)),
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )
        self.recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            initial_value=self.recurrent_init(shape=(self.units, 3 * self.units)),
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )
        if self.use_bias:
            self.bias = self.add_variable(
                "bias",
                initial_value=self.bias_init(shape=(3 * self.units,)),
                device=self.device,
                shard_devices=self.shard_devices,
                dtype=self.dtype,
                trainable=self.trainable,
            )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def call(
        self,
        inputs,
        *,
        training: bool = False,
        mask=None,
        initial_state=None,
    ):
        if inputs.ndim != 3:
            raise ValueError("GRU expects 3-D inputs (batch, time, features).")

        if mask is not None:
            inputs = jnp.where(mask[..., None], inputs, 0.0)  # hard mask

        # Grab raw arrays once
        K = self.kernel.value
        R = self.recurrent_kernel.value
        B = self.bias.value if self.use_bias else None
        W_z, W_r, W_h = jnp.split(K, 3, axis=1)
        U_z, U_r, U_h = jnp.split(R, 3, axis=1)
        if B is not None:
            b_z, b_r, b_h = jnp.split(B, 3)
        else:
            b_z = b_r = b_h = 0.0

        batch_size = inputs.shape[0]
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.units), dtype=self.dtype)

        inputs_T = jnp.swapaxes(inputs, 0, 1)  # (time, batch, feat)

        def step(h_prev, x_t):
            z = self.recurrent_activation(x_t @ W_z + h_prev @ U_z + b_z)
            r = self.recurrent_activation(x_t @ W_r + h_prev @ U_r + b_r)
            h_hat = self.activation(x_t @ W_h + (r * h_prev) @ U_h + b_h)
            h_t = (1.0 - z) * h_prev + z * h_hat
            return h_t, h_t

        h_last, h_all = lax.scan(step, initial_state, inputs_T)
        outputs = jnp.swapaxes(h_all, 0, 1)  # (batch, time, units)

        main = outputs if self.return_sequences else h_last
        return (main, h_last) if self.return_state else main

    # ------------------------------------------------------------------ #
    # Pure functional variant
    # ------------------------------------------------------------------ #
    def functional_call(
        self,
        inputs,
        params,
        *,
        training: bool = False,
        mask=None,
        initial_state=None,
    ):
        if mask is not None:
            inputs = jnp.where(mask[..., None], inputs, 0.0)

        K = params["kernel"]
        R = params["recurrent_kernel"]
        B = params.get("bias", None)
        W_z, W_r, W_h = jnp.split(K, 3, axis=1)
        U_z, U_r, U_h = jnp.split(R, 3, axis=1)
        if B is not None:
            b_z, b_r, b_h = jnp.split(B, 3)
        else:
            b_z = b_r = b_h = 0.0

        batch_size = inputs.shape[0]
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.units), dtype=inputs.dtype)

        inputs_T = jnp.swapaxes(inputs, 0, 1)

        def step(h_prev, x_t):
            z = self.recurrent_activation(x_t @ W_z + h_prev @ U_z + b_z)
            r = self.recurrent_activation(x_t @ W_r + h_prev @ U_r + b_r)
            h_hat = self.activation(x_t @ W_h + (r * h_prev) @ U_h + b_h)
            h_t = (1.0 - z) * h_prev + z * h_hat
            return h_t, h_t

        h_last, h_all = lax.scan(step, initial_state, inputs_T)
        outputs = jnp.swapaxes(h_all, 0, 1)
        main = outputs if self.return_sequences else h_last
        return (main, h_last) if self.return_state else main

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def compute_output_shape(self, input_shape):
        b, t, _ = input_shape
        main = (b, t, self.units) if self.return_sequences else (b, self.units)
        return (main, (b, self.units)) if self.return_state else main

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                units=self.units,
                use_bias=self.use_bias,
                return_sequences=self.return_sequences,
                return_state=self.return_state,
            )
        )
        return cfg

    def __repr__(self):
        return (
            f"<GRU units={self.units}, "
            f"return_sequences={self.return_sequences}, "
            f"built={self.built}>"
        )
