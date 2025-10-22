"""Parameterization backend registry for OpenMM preparation."""

from .base import (
    ParameterizationBackend,
    ParameterizationResult,
    available_backends,
    get_backend,
    register_backend,
)

# Import side-effect modules to populate the registry.
from . import ambertools  # noqa: F401
from . import openff  # noqa: F401

__all__ = [
    "ParameterizationBackend",
    "ParameterizationResult",
    "available_backends",
    "get_backend",
    "register_backend",
]
