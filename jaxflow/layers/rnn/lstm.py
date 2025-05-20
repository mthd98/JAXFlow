import jax
import jax.numpy as jnp
from jax import lax
from jaxflow.layers.layer import Layer
from jaxflow.initializers.initializers import GlorotUniform, Orthogonal, Zeros

class LSTM(Layer):
    """Long‑Short Term Memory layer for **jaxflow** (Elman‑style sequence‑first).

    Input shape: *(batch, timesteps, features)* ➜ Output shape depends on
    `return_sequences`:

    * `False`: *(batch, units)* (last hidden state)
    * `True`:  *(batch, timesteps, units)* (full sequence)

    Parameters
    ----------
    units : int
        Dimensionality of the hidden state *h_t* / cell state *c_t*.
    activation : callable, default ``jax.nn.tanh``
        Activation used for candidate cell (*g*) & output modulation.
    recurrent_activation : callable, default ``jax.nn.sigmoid``
        Activation for gates (input, forget, output).
    use_bias : bool, default ``True``
        If ``True`` add learned bias to the linear transformations.
    return_sequences : bool, default ``False``
        Whether to return the full sequence or only the last hidden state.
    return_state : bool, default ``False``
        If ``True`` also return the final ``(h_T, c_T)`` tuple.
    kernel_initializer, recurrent_initializer, bias_initializer : Initializer
        classes / callables accepting ``seed`` & ``dtype``.
    dtype : jnp.dtype, default ``jnp.float32``.
    seed : int | None
        Base RNG seed; if ``None`` choose randomly.
    """

    def __init__(
        self,
        units: int,
        *,
        name=None,
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
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Default initializers
        kernel_initializer = kernel_initializer or GlorotUniform
        recurrent_initializer = recurrent_initializer or Orthogonal
        bias_initializer = bias_initializer or Zeros

        self.kernel_initializer = kernel_initializer(seed=seed, dtype=dtype) if callable(kernel_initializer) else kernel_initializer
        self.recurrent_initializer = (
            recurrent_initializer(seed=seed, dtype=dtype) if callable(recurrent_initializer) else recurrent_initializer
        )
        self.bias_initializer = bias_initializer(seed=seed, dtype=dtype) if callable(bias_initializer) else bias_initializer

        self.device = device
        self.shard_devices = shard_devices
        self.dtype = dtype
        self.seed = seed

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("LSTM expects (batch, time, features) inputs. Got shape %s" % (input_shape,))
        features = input_shape[-1]

        # kernels: (features, 4 * units) and (units, 4 * units)
        k_shape = (features, 4 * self.units)
        rk_shape = (self.units, 4 * self.units)

        self.kernel = self.add_variable(
            "kernel",
            initial_value=self.kernel_initializer(shape=k_shape),
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )

        self.recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            initial_value=self.recurrent_initializer(shape=rk_shape),
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )

        if self.use_bias:
            self.bias = self.add_variable(
                "bias",
                initial_value=self.bias_initializer(shape=(4 * self.units,)),
                device=self.device,
                shard_devices=self.shard_devices,
                dtype=self.dtype,
                trainable=self.trainable,
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _step(self, carry, x_t):
        """Single‑timestep update used by `lax.scan`.``carry`` = (h, c)."""
        h_prev, c_prev = carry
        z = x_t @ self.kernel + h_prev @ self.recurrent_kernel
        if self.use_bias:
            z = z + self.bias
        i, f, g, o = jnp.split(z, 4, axis=-1)
        i = self.recurrent_activation(i)
        f = self.recurrent_activation(f)
        o = self.recurrent_activation(o)
        g = self.activation(g)
        c = f * c_prev + i * g
        h = o * self.activation(c)
        return (h, c), h

    def call(self, inputs, training=False, mask=None, initial_state=None):
        # inputs: (batch, time, features)
        batch_size = inputs.shape[0]
        if initial_state is None:
            h0 = jnp.zeros((batch_size, self.units), dtype=self.dtype)
            c0 = jnp.zeros((batch_size, self.units), dtype=self.dtype)
        else:
            h0, c0 = initial_state
        (h_final, c_final), h_seq = lax.scan(self._step, (h0, c0), inputs.swapaxes(0, 1))
        h_seq = h_seq.swapaxes(0, 1)  # back to (batch, time, units)

        if self.return_sequences and self.return_state:
            return h_seq, (h_final, c_final)
        elif self.return_sequences:
            return h_seq
        elif self.return_state:
            return h_final, (h_final, c_final)
        else:
            return h_final

    # ------------------------------------------------------------------
    # Functional
    # ------------------------------------------------------------------
    def functional_call(self, inputs, params, *, initial_state=None, training=False, mask=None):
        kernel = params["kernel"]
        recurrent_kernel = params["recurrent_kernel"]
        bias = params.get("bias", None)

        def step(carry, x_t):
            h_prev, c_prev = carry
            z = x_t @ kernel + h_prev @ recurrent_kernel
            if bias is not None:
                z = z + bias
            i, f, g, o = jnp.split(z, 4, axis=-1)
            i = self.recurrent_activation(i)
            f = self.recurrent_activation(f)
            o = self.recurrent_activation(o)
            g = self.activation(g)
            c = f * c_prev + i * g
            h = o * self.activation(c)
            return (h, c), h

        batch_size = inputs.shape[0]
        if initial_state is None:
            h0 = jnp.zeros((batch_size, self.units), dtype=self.dtype)
            c0 = jnp.zeros((batch_size, self.units), dtype=self.dtype)
        else:
            h0, c0 = initial_state

        (h_final, c_final), h_seq = lax.scan(step, (h0, c0), inputs.swapaxes(0, 1))
        h_seq = h_seq.swapaxes(0, 1)

        if self.return_sequences and self.return_state:
            return h_seq, (h_final, c_final)
        elif self.return_sequences:
            return h_seq
        elif self.return_state:
            return h_final, (h_final, c_final)
        else:
            return h_final

    # ------------------------------------------------------------------
    # Helpers / metadata
    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        return (input_shape[0], self.units)

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "units": self.units,
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "use_bias": self.use_bias,
            }
        )
        return base

    def __repr__(self):
        cfg = self.get_config()
        return (
            f"<LSTM units={cfg['units']}, return_sequences={cfg['return_sequences']}, "
            f"built={self.built}>"
        )


#test
lstm = LSTM(units=10, return_sequences=False)
lstm.build((None, None, 5))
x = jnp.ones((2, 3, 5))
lstm(x).shape