import jax
import jax.numpy as jnp
from jax import lax
from jaxflow.layers.layer import Layer
from jaxflow.initializers.initializers import GlorotUniform, Orthogonal, Zeros



class GRU(Layer):
    """Gated Recurrent Unit layer for **jaxflow**.

    Input:  *(batch, timesteps, features)*  (N, T, F)
    Output: *(batch, units)* if `return_sequences=False`, otherwise *(batch, timesteps, units)*.

    Gates (Kyunghyun Cho et al., 2014):

    .. math::
        z_t &= \sigma(x_t W_{xz} + h_{t-1} W_{hz} + b_z) \\
        r_t &= \sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r) \\
        \tilde{h}_t &= \phi(x_t W_{xh} + (r_t * h_{t-1}) W_{hh} + b_h) \\
        h_t &= (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t

    where :math:`\sigma` is `recurrent_activation` (default **sigmoid**) and
    :math:`\phi` is `activation` (default **tanh**).
    """

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

        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.device = device
        self.shard_devices = shard_devices
        self.dtype = dtype
        self.seed = seed

        # Default initializers
        kernel_initializer = kernel_initializer or GlorotUniform
        recurrent_initializer = recurrent_initializer or Orthogonal
        bias_initializer = bias_initializer or Zeros

        # Instantiate
        self.kernel_initializer = (
            kernel_initializer(seed=seed, dtype=dtype)
            if callable(kernel_initializer)
            else kernel_initializer
        )
        self.recurrent_initializer = (
            recurrent_initializer(seed=seed, dtype=dtype)
            if callable(recurrent_initializer)
            else recurrent_initializer
        )
        self.bias_initializer = (
            bias_initializer(seed=seed, dtype=dtype)
            if callable(bias_initializer)
            else bias_initializer
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "GRU expects inputs of shape (batch, timesteps, features)."
            )
        in_features = input_shape[-1]

        # Concatenate kernels for z, r, h~ → shape (F, 3U)
        kernel_shape = (in_features, 3 * self.units)
        recurrent_shape = (self.units, 3 * self.units)

        k_init = self.kernel_initializer(shape=kernel_shape)
        r_init = self.recurrent_initializer(shape=recurrent_shape)

        self.kernel = self.add_variable(
            "kernel",
            initial_value=k_init,
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )
        self.recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            initial_value=r_init,
            device=self.device,
            shard_devices=self.shard_devices,
            dtype=self.dtype,
            trainable=self.trainable,
        )

        if self.use_bias:
            b_init = self.bias_initializer(shape=(3 * self.units,))
            self.bias = self.add_variable(
                "bias",
                initial_value=b_init,
                device=self.device,
                shard_devices=self.shard_devices,
                dtype=self.dtype,
                trainable=self.trainable,
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _step(self, carry, x_t):
        """One GRU timestep; used inside lax.scan."""
        h_prev = carry

        # Split weights: z, r, h~
        W_z, W_r, W_h = jnp.split(self.kernel.value, 3, axis=1)
        U_z, U_r, U_h = jnp.split(self.recurrent_kernel.value, 3, axis=1)

        if self.use_bias:
            b_z, b_r, b_h = jnp.split(self.bias.value, 3)
        else:
            b_z = b_r = b_h = 0.0

        z_t = self.recurrent_activation(x_t @ W_z + h_prev @ U_z + b_z)
        r_t = self.recurrent_activation(x_t @ W_r + h_prev @ U_r + b_r)
        h_hat = self.activation(x_t @ W_h + (r_t * h_prev) @ U_h + b_h)
        h_t = (1.0 - z_t) * h_prev + z_t * h_hat
        return h_t, h_t  # carry, y_out (y==h for vanilla GRU)

    def call(self, inputs, training=False, mask=None, initial_state=None):  # noqa: D401
        if not self.built:
            self.build(inputs.shape)
        batch_size = inputs.shape[0]
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.units), dtype=self.dtype)

        last_state, outputs = lax.scan(self._step, initial_state, inputs.swapaxes(0, 1))
        # lax.scan outputs shape (T, N, U) – transpose back
        outputs = outputs.swapaxes(0, 1)

        if self.return_sequences:
            result = outputs
        else:
            result = outputs[:, -1, :]

        if self.return_state:
            return result, last_state
        return result

    # ------------------------------------------------------------------
    # Functional call
    # ------------------------------------------------------------------
    def functional_call(self, inputs, params, *, initial_state=None, training=False, mask=None):
        kernel = params["kernel"]
        recurrent_kernel = params["recurrent_kernel"]
        bias = params.get("bias", None)

        def step(carry, x_t):
            h_prev = carry
            W_z, W_r, W_h = jnp.split(kernel, 3, axis=1)
            U_z, U_r, U_h = jnp.split(recurrent_kernel, 3, axis=1)
            if bias is not None:
                b_z, b_r, b_h = jnp.split(bias, 3)
            else:
                b_z = b_r = b_h = 0.0
            z_t = self.recurrent_activation(x_t @ W_z + h_prev @ U_z + b_z)
            r_t = self.recurrent_activation(x_t @ W_r + h_prev @ U_r + b_r)
            h_hat = self.activation(x_t @ W_h + (r_t * h_prev) @ U_h + b_h)
            h_t = (1.0 - z_t) * h_prev + z_t * h_hat
            return h_t, h_t

        batch_size = inputs.shape[0]
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.units), dtype=inputs.dtype)

        last_state, outputs = lax.scan(step, initial_state, inputs.swapaxes(0, 1))
        outputs = outputs.swapaxes(0, 1)

        result = outputs if self.return_sequences else outputs[:, -1, :]
        if self.return_state:
            return result, last_state
        return result

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        batch, timesteps, _ = input_shape
        if self.return_sequences:
            return (batch, timesteps, self.units)
        return (batch, self.units)

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
            }
        )
        return base

    def __repr__(self):
        cfg = self.get_config()
        return (
            f"<GRU units={cfg['units']}, return_sequences={cfg['return_sequences']}, "
            f"built={self.built}>"
        )


