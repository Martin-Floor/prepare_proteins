# prepare_proteins/MD/parameterization/__init__.py

from .base import (
    ParameterizationBackend,
    ParameterizationResult,
    available_backends,
    get_backend,
    register_backend,
)

# Always import other backends that don’t need optional deps
from . import ambertools  # noqa: F401

# Try to import the OpenFF backend (which has optional deps)
try:
    # If your OpenFF backend uses @register_backend, just importing the module registers it.
    from . import openff  # noqa: F401
except Exception as exc:
    # OpenFF (and/or its dependencies) unavailable: register a placeholder backend.
    @register_backend
    class _UnavailableOpenFFBackend(ParameterizationBackend):
        name = "openff"
        input_format = "amber"

        def __init__(self, _exc=exc, **options):  # type: ignore[override]
            raise RuntimeError(
                "The 'openff' parameterization backend requires optional dependencies "
                "(openmm, openmmforcefields, openff-toolkit, parmed). "
                "Install them to enable this backend."
            ) from _exc  # type: ignore[name-defined]

__all__ = [
    "ParameterizationBackend",
    "ParameterizationResult",
    "available_backends",
    "get_backend",
    "register_backend",
]
