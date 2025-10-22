from typing import Any

from .base import ParameterizationBackend, ParameterizationResult, register_backend


@register_backend
class OpenFFBackend(ParameterizationBackend):
    """Placeholder OpenFF backend. Implementation to be provided separately."""

    name = "openff"
    input_format = "openff"

    def prepare_model(self, openmm_md, parameters_folder: str, **kwargs: Any) -> ParameterizationResult:
        raise NotImplementedError(
            "OpenFF parameterization backend is not implemented yet. "
            "Please select 'ambertools' or provide an implementation."
        )
