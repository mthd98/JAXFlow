import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

class Variable:
    def __init__(self,
                 initial_value=None,
                 trainable=True,
                 name=None,
                 variable_def=None,
                 dtype=None,
                 shape=None,
                 device='auto',
                 shard_devices=None,
                 sharding=None,
                 memory_kind=None):
        """
        Args:
            initial_value: The initial value (Python or JAX array).
            trainable: Whether the variable is trainable.
            name: Variable name.
            variable_def: Optional variable definition metadata.
            dtype: Data type; if provided, initial_value is cast to this type.
            shape: Expected shape (required if initial_value is None).
            device: A string indicating the device ('auto', 'cpu', 'gpu', or 'tpu')
                    if no sharding is used.
            shard_devices: Controls sharding across multiple devices. It can be:
                - None: the variable is placed on a single device.
                - A list of JAX devices: the variable is sharded across these devices.
                - "auto": automatically selects all devices that match the platform
                          of the device chosen by the `device` parameter.
            sharding: A full JAX sharding specification (e.g. a NamedSharding) to use.
            memory_kind: If sharding is provided, you can override its memory kind
                         (e.g. 'pinned_host' or 'device') via with_memory_kind().
        """
        # Create initial_value if not provided.
        if initial_value is None:
            if shape is None:
                raise ValueError("Either initial_value or shape must be provided")
            dtype = dtype or jnp.float32
            initial_value = jnp.zeros(shape, dtype=dtype)
        else:
            initial_value = jnp.array(initial_value, dtype=dtype) if dtype is not None else jnp.array(initial_value)

        # Validate shape if provided.
        if shape is not None and initial_value.shape != shape:
            raise ValueError(f"Provided shape {shape} does not match initial value shape {initial_value.shape}")

        self.trainable = trainable
        self.name = name
        self.variable_def = variable_def
        self.shape = initial_value.shape
        self.dtype = initial_value.dtype

        # Option 1: Use a provided full sharding specification.
        if sharding is not None:
            if memory_kind is not None and hasattr(sharding, 'with_memory_kind'):
                sharding = sharding.with_memory_kind(memory_kind)
            self.sharding = sharding
            self.value = jax.device_put(initial_value, device=sharding)
        # Option 2: Use shard_devices for sharding across multiple devices.
        elif shard_devices is not None:
            # If shard_devices is "auto", select devices based on the chosen single device's platform.
            if isinstance(shard_devices, str) and shard_devices.lower() == "auto":
                auto_device = self._select_device(device)
                selected_devices = [d for d in jax.devices() if d.platform == auto_device.platform]
            elif isinstance(shard_devices, list):
                selected_devices = shard_devices
            else:
                raise ValueError("shard_devices must be either 'auto', a list of devices, or None")

            # Ensure the leading axis can be evenly split among the devices.
            shard_count = len(selected_devices)
            if initial_value.shape[0] % shard_count != 0:
                raise ValueError("The leading dimension of initial_value must be divisible by the number of shard devices")
            shards = jnp.split(initial_value, shard_count)
            self.value = jax.device_put_sharded(shards, devices=selected_devices)
            self._shard_devices = selected_devices  # Save for later use in the property.
        # Option 3: Single device placement.
        else:
            self._single_device = self._select_device(device)
            self.value = jax.device_put(initial_value, device=self._single_device)

    def _select_device(self, device):
        """
        Select a JAX device based on the provided device string.
        If 'auto', prefers GPU, then TPU, then CPU.
        """
        devices = jax.devices()
        if device == "auto":
            # Prefer GPU if available.
            gpu_devices = [d for d in devices if d.platform == "gpu"]
            if gpu_devices:
                return gpu_devices[0]
            # Then try TPU.
            tpu_devices = [d for d in devices if d.platform == "tpu"]
            if tpu_devices:
                return tpu_devices[0]
            # Fall back to CPU.
            cpu_devices = [d for d in devices if d.platform == "cpu"]
            if cpu_devices:
                return cpu_devices[0]
            return devices[0]
        else:
            selected = [d for d in devices if d.platform == device.lower()]
            if not selected:
                raise ValueError(f"No devices found for platform: {device}")
            return selected[0]

    @property
    def device(self):
        """
        Returns the list of devices used by the variable.
        If the variable is not sharded, returns a one-item list with the single device.
        """
        if hasattr(self, 'sharding') and self.sharding is not None:
            # If using a full sharding specification, try to extract devices from the mesh.
            try:
                return list(self.sharding.mesh.devices)
            except AttributeError:
                # Fall back to a string representation if not available.
                return [str(self.sharding)]
        elif hasattr(self, '_shard_devices'):
            return self._shard_devices
        else:
            return [self._single_device]

    def assign(self, new_value):
        """Assign a new value to the variable, preserving its device/sharding layout."""
        new_value = jnp.array(new_value, dtype=self.dtype)
        if new_value.shape != self.value.shape:
            raise ValueError("New value shape must match the variable's shape")

        if hasattr(self, 'sharding') and self.sharding is not None:
            self.value = jax.device_put(new_value, device=self.sharding)
        elif hasattr(self, '_shard_devices'):
            shard_count = len(self._shard_devices)
            if new_value.shape[0] % shard_count != 0:
                raise ValueError("The leading dimension of new value must be divisible by the number of shard devices")
            shards = jnp.split(new_value, shard_count)
            self.value = jax.device_put_sharded(shards, devices=self._shard_devices)
        else:
            self.value = jax.device_put(new_value, device=self._single_device)

    def read_value(self):
        """Return the current value (a sharded or single jax.Array)."""
        return self.value

    def numpy(self):
        """Gather and return the underlying array as a host array."""
        return jax.device_get(self.value)

    # ------------------------------------------------------------------
    # Tell JAX (and NumPy) how to turn a Variable into an array
    # ------------------------------------------------------------------
    def __jax_array__(self, dtype=None):
        """Let JAX treat Variable as a jax.Array / tracer input."""
        return self.value.astype(dtype) if dtype is not None else self.value

    def __array__(self, dtype=None):
        """Let NumPy convert Variable → np.ndarray when needed."""
        host = jax.device_get(self.value)      # device→host copy if necessary
        return host.astype(dtype) if dtype is not None else host

     # -------------- ARITHMETIC OVERLOADS --------------

    # +  and  r+
    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    # -  and  r-
    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    # *  and  r*
    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    # /  and  r/
    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    # % (mod)  and  r%
    def __mod__(self, other):
        return self.value % other

    def __rmod__(self, other):
        return other % self.value

    # @  and  r@  (matrix multiply)
    def __matmul__(self, other):
        """
        This uses jnp.matmul under the hood if you do my_var @ other_var
        or my_var @ jnp.array(...).
        """
        return self.value @ other

    def __rmatmul__(self, other):
        """
        Called if the left operand doesn't implement __matmul__.
        """
        return other @ self.value

    # You can add more, e.g. __pow__, __floordiv__, etc.

    # -------------- ATTRIBUTE FORWARDING (Optional) --------------
    def __getattr__(self, name):
        """
        If you want methods like my_var.mean() or my_var.sum() to work,
        forward attribute lookups to the underlying jax array.
        """
        return getattr(self.value, name)

    def __repr__(self):
        full_value = jax.device_get(self.value)
        name_str = f"{self.name}:0" if self.name is not None else "Variable:0"
        if hasattr(self, 'sharding') and self.sharding is not None:
            try:
                device_list = list(self.sharding.mesh.devices)
                device_str = f"sharded over {device_list}"
            except AttributeError:
                device_str = f"sharded with {self.sharding}"
        elif hasattr(self, '_shard_devices'):
            device_str = f"sharded over {self._shard_devices}"
        else:
            device_str = str(self._single_device)
        return (f"<jaxflow.Variable '{name_str}' shape={self.shape} dtype={self.dtype}, "
                f"device={device_str}, numpy=\n{full_value}>")

# Example usage:
if __name__ == "__main__":
    # Option A: Single device variable (automatically selects GPU/TPU/CPU)
    initial_val = jnp.arange(9).reshape((3, 3))
    var_single = Variable(initial_value=initial_val, name="weight", device='auto')
    print("Single-device variable:")
    print(var_single)
    print("Devices:", var_single.device)

    # Option B: Sharding using shard_devices set to "auto"
    # This will automatically select all devices that match the platform determined by 'device'.
    var_sharded_auto = Variable(initial_value=jnp.arange(16).reshape((4, 4)),
                                name="kernel",
                                device='cpu',   # or 'gpu' or 'tpu'
                                shard_devices="auto")
    print("\nVariable sharded automatically over devices of type 'cpu':")
    print(var_sharded_auto)
    print("Devices:", var_sharded_auto.device)

    # Option C: Sharding using shard_devices as a list.
    # Here we use CPU devices for demonstration (most systems have at least one).
    cpu_devices = [d for d in jax.devices() if d.platform == "cpu"]
    # Make sure the shape's leading axis is divisible by number of devices.
    if cpu_devices and (initial_val.shape[0] % len(cpu_devices) == 0):
        var_sharded_list = Variable(initial_value=jnp.arange(16).reshape((4, 4)),
                                    name="bias",
                                    shard_devices=cpu_devices)
        print("\nVariable sharded over provided device list:")
        print(var_sharded_list)
    else:
        print("\nNo suitable CPU devices found or shape mismatch for list-based sharding.")

