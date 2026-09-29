from typing import Any

from .base import ParameterizationBackend, ParameterizationResult, register_backend


@register_backend
class AmberToolsBackend(ParameterizationBackend):
    """Parameterization backend leveraging the existing AmberTools workflow."""

    name = "ambertools"
    input_format = "amber"

    def prepare_model(self, openmm_md, parameters_folder: str, **kwargs: Any) -> ParameterizationResult:
        kwargs.pop("verbose", None)
        # Backend options come from setUpOpenMMSimulations(parameterization_options=...).
        if "gaff_version" in self.options:
            kwargs.setdefault("gaff_version", self.options["gaff_version"])
        openmm_md.parameterizePDBLigands(parameters_folder, **kwargs)
        result = self.describe_model(openmm_md)
        mcpb_config = getattr(openmm_md, "mcpb_config", None)
        if mcpb_config:
            result.metadata["mcpb_config"] = mcpb_config
        return result
