import jax.numpy as jnp
from jax import lax
from jaxflow.layers.layer import Layer
from jaxflow.initializers import GlorotUniform, Zeros,Orthogonal

class RNN(Layer):
    """Vanilla (Elman) recurrent layer for **jaxflow**.

    This is the simplest recurrent block – a single hidden‑state update per
    timestep:

    ``h_t = activation(x_t @ W_xh + h_{t-1} @ W_hh + b)``

    Parameters mirror Keras‐style RNNs and the previously defined Conv* layers
    so you can drop it into the same *Model* workflow.

    Args
    ----
    units : int
        Dimensionality of the hidden state and output space.
    activation : callable, optional
        Non‑linearity.  Defaults to ``jax.nn.tanh``.
    use_bias : bool, optional
        Include an additive bias term.  Default ``True``.
    return_sequences : bool, optional
        If ``True`` return the full sequence *(batch, time, units)*; otherwise
        just the last hidden state *(batch, units)*.  Default ``False``.
    return_state : bool, optional
        If ``True`` also return the final hidden state in addition to the
        sequence/last output.  Default ``False``.
    kernel_initializer, recurrent_initializer, bias_initializer : callables, optional
        Initializer factories. Each must accept ``seed`` and ``dtype`` and
        return a **jax.numpy** array when called with ``shape``.  Defaults are
        *Glorot(U)* for input kernel, *Orthogonal* for recurrent kernel and
        *Zeros* for bias.
    seed : int | None, optional
        Base RNG seed.
    dtype : jnp.dtype, optional
        Parameter dtype – default ``jnp.float32``.
    device, shard_devices, ... :
        Passed straight through to :class:`jaxflow.core.variable.Variable`.
    """

    def __init__(
        self,
        units: int,
        *,
        name=None,
        device="auto",
        shard_devices=None,
        dtype=jnp.float32,
        trainable=True,
        activation=None,
        use_bias: bool = True,
        return_sequences: bool = False,
        return_state: bool = False,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable)
        self.units = units
        self.activation = activation or jnp.tanh
        self.use_bias = use_bias
        self.return_sequences = return_sequences
        self.return_state = return_state

        if kernel_initializer is None:
            kernel_initializer = GlorotUniform
        if recurrent_initializer is None:
            recurrent_initializer = Orthogonal
        if bias_initializer is None:
            bias_initializer = Zeros

        self.kernel_initializer = kernel_initializer(seed=seed, dtype=dtype) if callable(kernel_initializer) else kernel_initializer
        self.recurrent_initializer = recurrent_initializer(seed=seed, dtype=dtype) if callable(recurrent_initializer) else recurrent_initializer
        self.bias_initializer = bias_initializer(seed=seed, dtype=dtype) if callable(bias_initializer) else bias_initializer

        # Store placeholders for compatibility – currently unused
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint

        self.device = device
        self.shard_devices = shard_devices
        self.dtype = dtype
        self.seed = seed

    # ------------------------------------------------------------------
    # Build & call
    # ------------------------------------------------------------------

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "SimpleRNN expects inputs shaped (batch, time, features). "
                f"Got {input_shape}."
            )
        _, _, in_features = input_shape

        # Weight: input→hidden  (features, units)
        W_xh_init = self.kernel_initializer(shape=(in_features, self.units))
        self.W_xh = self.add_variable(
            name="kernel",
            initial_value=W_xh_init,
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )

        # Recurrent weight: hidden→hidden  (units, units)
        W_hh_init = self.recurrent_initializer(shape=(self.units, self.units))
        self.W_hh = self.add_variable(
            name="recurrent_kernel",
            initial_value=W_hh_init,
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )

        if self.use_bias:
            b_init = self.bias_initializer(shape=(self.units,))
            self.bias = self.add_variable(
                name="bias",
                initial_value=b_init,
                device=self.device,
                shard_devices=self.shard_devices,
                dtype=self.dtype,
                trainable=self.trainable,
            )

    def _step(self, h_prev, x_t):
        # x_t ... (batch, features)
        h_linear = x_t @ self.W_xh + h_prev @ self.W_hh
        if self.use_bias:
            h_linear = h_linear + self.bias
        h_t = self.activation(h_linear)
        return h_t, h_t  # carry, output

    def call(self, inputs, *, training=False, mask=None, initial_state=None):
        # inputs: (batch, time, features)
        if inputs.ndim != 3:
            raise ValueError("SimpleRNN expects 3‑D inputs (batch, time, features).")
        batch_size = inputs.shape[0]

        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.units), dtype=self.dtype)

        # swap time & batch for lax.scan: (time, batch, features)
        inputs_T = jnp.swapaxes(inputs, 0, 1)

        h_last, h_all = lax.scan(self._step, initial_state, inputs_T)
        # h_all: (time, batch, units)
        outputs = jnp.swapaxes(h_all, 0, 1)  # (batch, time, units)

        if self.return_sequences:
            main_out = outputs
        else:
            main_out = h_last  # already (batch, units)

        if self.return_state:
            return main_out, h_last
        return main_out

    # ------------------------------------------------------------------
    # Functional (pure) call
    # ------------------------------------------------------------------

    def functional_call(self, inputs, params, *, training=False, mask=None, initial_state=None):
        W_xh = params["kernel"]
        W_hh = params["recurrent_kernel"]
        bias = params.get("bias", None)

        if initial_state is None:
            batch_size = inputs.shape[0]
            initial_state = jnp.zeros((batch_size, self.units), dtype=self.dtype)

        inputs_T = jnp.swapaxes(inputs, 0, 1)

        def step(h_prev, x_t):
            h_lin = x_t @ W_xh + h_prev @ W_hh
            if bias is not None:
                h_lin = h_lin + bias
            h_t = self.activation(h_lin)
            return h_t, h_t

        h_last, h_all = lax.scan(step, initial_state, inputs_T)
        outputs = jnp.swapaxes(h_all, 0, 1)

        main_out = outputs if self.return_sequences else h_last
        return (main_out, h_last) if self.return_state else main_out

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def compute_output_shape(self, input_shape):
        batch, time, _ = input_shape
        main = (batch, time, self.units) if self.return_sequences else (batch, self.units)
        if self.return_state:
            return (main, (batch, self.units))
        return main

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
        })
        return cfg

    def __repr__(self):
        return f"<SimpleRNN units={self.units}, return_sequences={self.return_sequences}, built={self.built}>"


