from typing import Any

from .base import ParameterizationBackend, ParameterizationResult, register_backend


@register_backend
class AmberToolsBackend(ParameterizationBackend):
    """Parameterization backend leveraging the existing AmberTools workflow."""

    name = "ambertools"
    input_format = "amber"

    def prepare_model(self, openmm_md, parameters_folder: str, **kwargs: Any) -> ParameterizationResult:
        openmm_md.parameterizePDBLigands(parameters_folder, **kwargs)
        return self.describe_model(openmm_md)
