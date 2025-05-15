"""
JAXFlow
=======

A lightweight, Flax-style neural-network library built on JAX.
"""

from __future__ import annotations
from importlib import metadata as _metadata
from typing import TYPE_CHECKING as _TYPE_CHECKING

# ---------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------
try:
    __version__: str = _metadata.version("jaxflow")
except _metadata.PackageNotFoundError:  # editable install
    __version__ = "0.0.3.dev0"

# Clean up namespace
del _metadata

# ---------------------------------------------------------------------
# Lazy-loaded sub-packages (heavy deps only on access)
# ---------------------------------------------------------------------
import importlib, types as _types

__lazy_subpackages = {
    "activations",
    "callbacks",
    "core",
    "gradient",
    "initializers",
    "layers",
    "losses",
    "math",
    "metrics",
    "models",
    "nn",
    "optimizers",
    "random",
    "regularizers",
}

def __getattr__(name: str) -> _types.ModuleType:
    if name in __lazy_subpackages:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # cache on first access
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ---------------------------------------------------------------------
# Help static type checkers know about our lazy modules
# ---------------------------------------------------------------------
if _TYPE_CHECKING:
    from . import (
        activations,
        callbacks,
        core,
        gradient,
        initializers,
        layers,
        losses,
        math,
        metrics,
        models,
        nn,
        optimizers,
        random,
        regularizers,
    )

# ---------------------------------------------------------------------
# What shows up on `from jaxflow import *`
# ---------------------------------------------------------------------
__all__ = [
    *sorted(__lazy_subpackages),
    "__version__",
]
