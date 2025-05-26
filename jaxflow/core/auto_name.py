
# jaxflow/utils/naming.py
from collections import defaultdict
from threading import Lock
from typing import Optional, Dict


class AutoNameMixin:
    """
    Mixin for automatic, collision-free name generation across subclasses.

    `AutoNameMixin` provides a reusable mechanism for generating unique, 
    class-scoped names for all objects that inherit from it. It is thread-safe 
    and ensures that each instance receives a unique name, even in parallel or 
    asynchronous environments.

    Key Features:
        - Globally unique auto-naming across all subclasses (e.g., Layer, Optimizer).
        - Class-level counters to ensure names like `Layer_1`, `Layer_2`, etc.
        - Thread-safe implementation using a global lock.
        - Counter reset utility for unit tests and notebook sessions.

    Methods:
        auto_name(provided: Optional[str] = None) -> str:
            Returns the provided name if not None, else generates a unique
            name in the format `<ClassName>_<N>`.

        reset_all_counters():
            Class method. Resets all internal counters. Useful for unit
            tests or interactive notebook restarts.

    Example:
        >>> class CustomObject(AutoNameMixin):
        ...     def __init__(self, name=None):
        ...         self.name = self.auto_name(name)
        ...
        >>> a = CustomObject()
        >>> b = CustomObject()
        >>> c = CustomObject(name="special")
        >>> print(a.name)  # CustomObject_1
        >>> print(b.name)  # CustomObject_2
        >>> print(c.name)  # special

    Attributes:
        _counters (Dict[str, int]): Class-level counters for name suffixes.
        _lock (Lock): Class-level lock for thread safety.
    """


    _counters: Dict[str, int] = defaultdict(int)
    _lock: Lock = Lock()

    @classmethod
    def _next_suffix(cls) -> int:
        with AutoNameMixin._lock:
            AutoNameMixin._counters[cls.__name__] += 1
            return AutoNameMixin._counters[cls.__name__]

    @classmethod
    def reset_all_counters(cls):
        """Utility for unit tests or notebook restarts."""
        with AutoNameMixin._lock:
            AutoNameMixin._counters.clear()

    # -----------------------------------------------------------------
    # Public helper
    # -----------------------------------------------------------------
    def auto_name(self, provided: Optional[str] = None) -> str:
        """
        Return `provided` if not None, else generate `<ClassName>_<N>`.
        """
        return provided or f"{self.__class__.__name__}_{self._next_suffix()}"