"""GAFF version helpers shared by the AmberTools/tleap setup paths (no OpenMM dependency)."""
import os


_GAFF_VERSIONS = ("gaff", "gaff2")


def _check_gaff_version(gaff_version):
    """Validate a GAFF version name ('gaff' or 'gaff2') and return it."""
    if gaff_version not in _GAFF_VERSIONS:
        raise ValueError(f"gaff_version must be one of {_GAFF_VERSIONS}, got {gaff_version!r}.")
    return gaff_version


def _gaff_leaprc(gaff_version):
    """tleap force-field file for the GAFF version."""
    return "leaprc.gaff2" if _check_gaff_version(gaff_version) == "gaff2" else "leaprc.gaff"


def _antechamber_atom_type_flag(gaff_version):
    """antechamber atom-type option; empty for GAFF, keeping the default commands unchanged."""
    return "-at gaff2 " if _check_gaff_version(gaff_version) == "gaff2" else ""


def _parmchk2_version_flag(gaff_version):
    """parmchk2 parameter-set option; empty for GAFF, keeping the default commands unchanged."""
    return "-s 2 " if _check_gaff_version(gaff_version) == "gaff2" else ""


def _read_gaff_version_marker(path):
    """GAFF version recorded in a <RES>.gaff_version file, or None when there is none."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return handle.read().strip() or None


def _tleap_gaff_version(tleap_path):
    """GAFF version a tleap input sources ('gaff', 'gaff2'), or None if it sources neither or is absent."""
    if not os.path.exists(tleap_path):
        return None
    with open(tleap_path) as handle:
        sources = {line.split()[1] for line in handle
                   if line.startswith("source ") and len(line.split()) > 1}
    if "leaprc.gaff2" in sources:
        return "gaff2"
    if "leaprc.gaff" in sources:
        return "gaff"
    return None
