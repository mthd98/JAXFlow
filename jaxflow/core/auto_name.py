
# jaxflow/utils/naming.py
from collections import defaultdict
from threading import Lock
from typing import Optional, Dict


class AutoNameMixin:
    """
    Re-usable “give-me-a-unique-name” helper.

    • Works across *all* subclasses that mix it in.
    • Counter is global per concrete class (Layer, Conv2D, Adam …).
    • Thread-safe: JAX async dispatch or dataloader threads won’t step
      on each other.
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