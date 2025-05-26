# jaxflow/layers/rnn.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from jaxflow.layers.layer import Layer
from jaxflow.initializers import GlorotUniform, Orthogonal, Zeros


class RNN(Layer):
    """
    Vanilla (Elman) recurrent layer for JAXFlow.

    Implements a simple recurrent neural network layer that maintains a hidden state
    across timesteps, following the classic Elman formulation:
        h_t = activation(x_t @ W_xh + h_{t-1} @ W_hh + b)

    Parameters
    ----------
    units : int
        Hidden/output dimensionality (number of units in the RNN).
    activation : callable, default jnp.tanh
        Activation function to use.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    return_sequences : bool, default False
        If True, returns the full sequence of outputs (batch, time, units).
        If False, returns only the last output (batch, units).
    return_state : bool, default False
        If True, returns a tuple (output, last_hidden_state).
    kernel_initializer : callable or Initializer, optional
        Initializer for the input kernel weights. Defaults to GlorotUniform.
    recurrent_initializer : callable or Initializer, optional
        Initializer for the recurrent kernel weights. Defaults to Orthogonal.
    bias_initializer : callable or Initializer, optional
        Initializer for the bias vector. Defaults to Zeros.
    dtype : jnp.dtype, default jnp.float32
        Data type for parameters.
    device : str, default "auto"
        Device for parameter placement ("auto", "cpu", "gpu", "tpu").
    shard_devices : list or str, optional
        Devices for parameter sharding (see Variable docs).
    seed : int, optional
        Random seed for parameter initialization.
    name : str, optional
        Layer name. If None, a unique name is generated.
    trainable : bool, default True
        Whether the parameters are trainable.

    Inputs:
        inputs (jnp.ndarray): 3D tensor of shape (batch, time, features).
        initial_state (jnp.ndarray, optional): Initial hidden state, shape (batch, units).
        mask (jnp.ndarray, optional): Boolean tensor of shape (batch, time), for input masking.

    Input shape:
        (batch_size, time_steps, features)

    Output shape:
        (batch_size, units) if return_sequences=False (default)
        (batch_size, time_steps, units) if return_sequences=True

        If return_state=True, returns a tuple: (output, last_hidden_state)

    Attributes:
        units (int): Number of output units.
        activation (callable): Activation function used.
        use_bias (bool): Whether bias is included.
        return_sequences (bool): Whether to return the full output sequence.
        return_state (bool): Whether to return the last hidden state.
        kernel (Variable): Input kernel weights.
        recurrent_kernel (Variable): Recurrent kernel weights.
        bias (Variable): Bias vector (if use_bias is True).
        built (bool): Whether the layer has been built.

    Example:
        ```python
        import jax.numpy as jnp
        from jaxflow.layers.rnn import RNN

        # Example input: batch of 8, sequence length 12, feature size 32
        x = jnp.ones((8, 12, 32))
        rnn = RNN(units=16, return_sequences=True)
        y = rnn(x)  # (8, 12, 16)
        ```

    Raises:
        ValueError: If input shape does not match (batch, time, features).

    Note:
        - The layer does not use a stateful mode; for stateful inference, manage the initial_state manually.
        - Supports masking for variable-length sequences.
        - Compatible with JAX JIT/vmap/pmap and functional API.
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
        activation=None,
        use_bias: bool = True,
        return_sequences: bool = False,
        return_state: bool = False,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        dtype=jnp.float32,
        device="auto",
        shard_devices=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable)

        # Public config
        self.units = int(units)
        self.activation = activation or jnp.tanh
        self.use_bias = bool(use_bias)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)

        # Parameter init factories
        kernel_initializer = kernel_initializer or GlorotUniform
        recurrent_initializer = recurrent_initializer or Orthogonal
        bias_initializer = bias_initializer or Zeros

        self.kernel_init = kernel_initializer(seed=seed, dtype=dtype) if callable(
            kernel_initializer
        ) else kernel_initializer
        self.recurrent_init = recurrent_initializer(seed=seed, dtype=dtype) if callable(
            recurrent_initializer
        ) else recurrent_initializer
        self.bias_init = bias_initializer(seed=seed, dtype=dtype) if callable(
            bias_initializer
        ) else bias_initializer

        # Misc bookkeeping
        self.dtype = dtype
        self.device = device
        self.shard_devices = shard_devices
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "RNN expects inputs shaped (batch, time, features); "
                f"got {input_shape}."
            )
        _, _, in_features = input_shape

        # Weights
        self.W_xh = self.add_variable(
            "kernel",
            initial_value=self.kernel_init(shape=(in_features, self.units)),
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )
        self.W_hh = self.add_variable(
            "recurrent_kernel",
            initial_value=self.recurrent_init(shape=(self.units, self.units)),
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )
        if self.use_bias:
            self.bias = self.add_variable(
                "bias",
                initial_value=self.bias_init(shape=(self.units,)),
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
            raise ValueError("RNN expects 3-D inputs (batch, time, features).")

        # Hard mask (optional) – zero-out masked timesteps before scan
        if mask is not None:
            inputs = jnp.where(mask[..., None], inputs, 0.0)

        # Grab raw arrays once, outside the scan body (perf!)
        W_xh = self.W_xh.value
        W_hh = self.W_hh.value
        b = self.bias.value if self.use_bias else None

        batch_size = inputs.shape[0]
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.units), dtype=self.dtype)

        inputs_T = jnp.swapaxes(inputs, 0, 1)  # (time, batch, feat)

        def step(h_prev, x_t):
            h_lin = x_t @ W_xh + h_prev @ W_hh
            if b is not None:
                h_lin = h_lin + b
            h_t = self.activation(h_lin)
            return h_t, h_t  # carry, output

        h_last, h_all = lax.scan(step, initial_state, inputs_T)
        outputs = jnp.swapaxes(h_all, 0, 1)  # (batch, time, units)

        main_out = outputs if self.return_sequences else h_last
        return (main_out, h_last) if self.return_state else main_out

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

        # Unpack arrays once
        W_xh = params["kernel"]
        W_hh = params["recurrent_kernel"]
        b = params.get("bias", None)

        batch_size = inputs.shape[0]
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.units), dtype=self.dtype)

        inputs_T = jnp.swapaxes(inputs, 0, 1)

        def step(h_prev, x_t):
            h_lin = x_t @ W_xh + h_prev @ W_hh
            if b is not None:
                h_lin = h_lin + b
            h_t = self.activation(h_lin)
            return h_t, h_t

        h_last, h_all = lax.scan(step, initial_state, inputs_T)
        outputs = jnp.swapaxes(h_all, 0, 1)
        main_out = outputs if self.return_sequences else h_last
        return (main_out, h_last) if self.return_state else main_out

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def compute_output_shape(self, input_shape):
        batch, time, _ = input_shape
        main = (batch, time, self.units) if self.return_sequences else (batch, self.units)
        return (main, (batch, self.units)) if self.return_state else main

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
            f"<RNN units={self.units}, "
            f"return_sequences={self.return_sequences}, "
            f"built={self.built}>"
        )
