# jaxflow/layers/lstm.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from jaxflow.layers.layer import Layer
from jaxflow.initializers import GlorotUniform, Orthogonal, Zeros


class LSTM(Layer):
    """
    Long Short-Term Memory (LSTM) layer for JAXFlow.

    Implements a standard LSTM recurrent layer, which learns long-range dependencies
    by using gates to control the flow of information. Follows Keras/PyTorch semantics
    for initialization, call signature, and output options.

    Args:
        units (int): Dimensionality of the output space (number of hidden units).
        activation (callable, optional): Activation function for the candidate cell state.
            Defaults to jax.nn.tanh.
        recurrent_activation (callable, optional): Activation function for the gates.
            Defaults to jax.nn.sigmoid.
        use_bias (bool, optional): Whether the layer uses a bias vector. Defaults to True.
        return_sequences (bool, optional): If True, returns the full sequence of outputs
            (batch, time, units). If False (default), returns only the last output (batch, units).
        return_state (bool, optional): If True, returns a tuple (output, (last_hidden, last_cell)).
        kernel_initializer (callable or Initializer, optional): Initializer for input kernel weights.
            Defaults to GlorotUniform.
        recurrent_initializer (callable or Initializer, optional): Initializer for recurrent kernel weights.
            Defaults to Orthogonal.
        bias_initializer (callable or Initializer, optional): Initializer for bias vectors.
            Defaults to Zeros.
        device (str, optional): Device for parameter placement ("auto", "cpu", "gpu", "tpu"). Defaults to "auto".
        shard_devices (list or str, optional): Devices for parameter sharding. See Variable docs.
        dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
        seed (int, optional): Random seed for parameter initialization.
        name (str, optional): Layer name. If None, a unique name is generated.
        trainable (bool, optional): Whether the parameters are trainable. Defaults to True.

    Inputs:
        inputs (jnp.ndarray): 3D tensor of shape (batch, time, features).
        initial_state (tuple of jnp.ndarray, optional): Initial hidden and cell states, each of shape (batch, units).
        mask (jnp.ndarray, optional): Boolean tensor with shape (batch, time), for masking input steps.

    Input shape:
        (batch_size, time_steps, features)

    Output shape:
        (batch_size, units) if return_sequences=False (default)
        (batch_size, time_steps, units) if return_sequences=True

        If return_state=True, returns a tuple: (output, (last_hidden_state, last_cell_state))

    Attributes:
        units (int): Dimensionality of the output space.
        activation (callable): Activation for candidate state.
        recurrent_activation (callable): Activation for gates.
        use_bias (bool): Whether bias is used.
        return_sequences (bool): Whether to return the full output sequence.
        return_state (bool): Whether to return the last hidden and cell states.
        kernel (Variable): Input kernel weights.
        recurrent_kernel (Variable): Recurrent kernel weights.
        bias (Variable): Bias vector (if use_bias is True).
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from jaxflow.layers.lstm import LSTM

        # Example input: batch of 8, sequence length 20, feature size 32
        x = jnp.ones((8, 20, 32))
        lstm = LSTM(units=16, return_sequences=True)
        y = lstm(x)  # (8, 20, 16)
        ```

    Raises:
        ValueError: If input shape does not match (batch, time, features).

    Note:
        - No explicit stateful mode; for stateful inference, manage initial_state manually.
        - Input and recurrent kernel matrices are concatenated internally for efficiency ([i, f, g, o] gates).
        - Compatible with JAX JIT/vmap/pmap for efficient batching and functional API usage.
    """


    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        units: int,
        *,
        name: str | None = None,
        trainable: bool = True,
        activation=jax.nn.tanh,
        recurrent_activation=jax.nn.sigmoid,
        use_bias: bool = True,
        return_sequences: bool = False,
        return_state: bool = False,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        device="auto",
        shard_devices=None,
        dtype=jnp.float32,
        seed=None,
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
            raise ValueError("LSTM expects (batch, time, features) inputs.")
        _, _, in_features = input_shape

        self.kernel = self.add_variable(
            "kernel",
            initial_value=self.kernel_init(shape=(in_features, 4 * self.units)),
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )
        self.recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            initial_value=self.recurrent_init(shape=(self.units, 4 * self.units)),
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )
        if self.use_bias:
            self.bias = self.add_variable(
                "bias",
                initial_value=self.bias_init(shape=(4 * self.units,)),
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
            raise ValueError("LSTM expects 3-D inputs (batch, time, features).")
        if mask is not None:
            inputs = jnp.where(mask[..., None], inputs, 0.0)

        # Capture raw arrays once
        K = self.kernel.value
        R = self.recurrent_kernel.value
        B = self.bias.value if self.use_bias else None
        (W_i, W_f, W_g, W_o) = jnp.split(K, 4, axis=1)
        (U_i, U_f, U_g, U_o) = jnp.split(R, 4, axis=1)
        if B is not None:
            b_i, b_f, b_g, b_o = jnp.split(B, 4)
        else:
            b_i = b_f = b_g = b_o = 0.0

        batch_size = inputs.shape[0]
        if initial_state is None:
            h0 = jnp.zeros((batch_size, self.units), dtype=self.dtype)
            c0 = jnp.zeros((batch_size, self.units), dtype=self.dtype)
        else:
            h0, c0 = initial_state

        inputs_T = jnp.swapaxes(inputs, 0, 1)  # (time, batch, feat)

        def step(carry, x_t):
            h_prev, c_prev = carry
            i = self.recurrent_activation(x_t @ W_i + h_prev @ U_i + b_i)
            f = self.recurrent_activation(x_t @ W_f + h_prev @ U_f + b_f)
            o = self.recurrent_activation(x_t @ W_o + h_prev @ U_o + b_o)
            g = self.activation(x_t @ W_g + h_prev @ U_g + b_g)
            c = f * c_prev + i * g
            h = o * self.activation(c)
            return (h, c), h

        (h_last, c_last), h_all = lax.scan(step, (h0, c0), inputs_T)
        outputs = jnp.swapaxes(h_all, 0, 1)  # (batch, time, units)

        main = outputs if self.return_sequences else h_last
        if self.return_state:
            return (main, (h_last, c_last))
        return main

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
        (W_i, W_f, W_g, W_o) = jnp.split(K, 4, axis=1)
        (U_i, U_f, U_g, U_o) = jnp.split(R, 4, axis=1)
        if B is not None:
            b_i, b_f, b_g, b_o = jnp.split(B, 4)
        else:
            b_i = b_f = b_g = b_o = 0.0

        batch_size = inputs.shape[0]
        if initial_state is None:
            h0 = jnp.zeros((batch_size, self.units), dtype=inputs.dtype)
            c0 = jnp.zeros((batch_size, self.units), dtype=inputs.dtype)
        else:
            h0, c0 = initial_state

        inputs_T = jnp.swapaxes(inputs, 0, 1)

        def step(carry, x_t):
            h_prev, c_prev = carry
            i = self.recurrent_activation(x_t @ W_i + h_prev @ U_i + b_i)
            f = self.recurrent_activation(x_t @ W_f + h_prev @ U_f + b_f)
            o = self.recurrent_activation(x_t @ W_o + h_prev @ U_o + b_o)
            g = self.activation(x_t @ W_g + h_prev @ U_g + b_g)
            c = f * c_prev + i * g
            h = o * self.activation(c)
            return (h, c), h

        (h_last, c_last), h_all = lax.scan(step, (h0, c0), inputs_T)
        outputs = jnp.swapaxes(h_all, 0, 1)
        main = outputs if self.return_sequences else h_last
        if self.return_state:
            return (main, (h_last, c_last))
        return main

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
            f"<LSTM units={self.units}, "
            f"return_sequences={self.return_sequences}, "
            f"built={self.built}>"
        )
