from __future__ import annotations

_OPENMM_IMPORT_ERROR = None
try:  # pragma: no cover - optional dependency
    import openmm  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    openmm = None  # type: ignore[assignment]
    _OPENMM_AVAILABLE = False
    _OPENMM_IMPORT_ERROR = exc
else:  # pragma: no cover - heavy optional dependency
    _OPENMM_AVAILABLE = True
    from openmm import *  # type: ignore
    from openmm.app import *  # type: ignore
    from openmm.unit import *  # type: ignore
    from openmm.vec3 import Vec3  # type: ignore

OPENMM_AVAILABLE = _OPENMM_AVAILABLE
OPENMM_IMPORT_ERROR = _OPENMM_IMPORT_ERROR

from Bio.PDB.Polypeptide import aa3

import io
import json
import numpy as np
from sys import stdout
from collections import defaultdict
from collections.abc import Mapping
from filecmp import cmp
import shutil
import os
import fileinput
from multiprocessing import cpu_count
import shlex
import warnings

from .parameterization.utils import extract_residue_subsystem
from .._resources import resource_listdir, resource_stream


def _ensure_openmm(feature: str = "OpenMM functionality") -> None:
    """
    Ensure the optional OpenMM dependency is available before executing OpenMM-dependent code.
    """
    if not OPENMM_AVAILABLE:
        raise ImportError(
            f"OpenMM is required to use {feature}. "
            "Install the 'openmm' package (e.g. `pip install openmm`) to enable this functionality."
        ) from OPENMM_IMPORT_ERROR

aa3 = list(aa3)+['HID', 'HIE', 'HIP', 'ASH', 'GLH', 'CYM', 'CYX', 'ACE', 'NME']
ions = ['MG', 'NA', 'CL', 'CU']
aa3 += ions

_SUPPORTED_MEMBRANE_LIPIDS = {
    "POPC": {
        "openmm_lipid_type": "POPC",
        "tleap_sources": ("oldff/leaprc.lipid17",),
        "skip_residue_names": ("POPC", "POP"),
    }
}
_SUPPORTED_MEMBRANE_ORIENTATION_MODES = {"as_is", "use_oriented_pdb"}

_MCPB_STRUCTURE_RESIDUE_ALIASES = {
    "ASH": ("ASP",),
    "CYM": ("CYS",),
    "CYX": ("CYS",),
    "GLH": ("GLU",),
    "HID": ("HIS",),
    "HIE": ("HIS",),
    "HIP": ("HIS",),
    "LYN": ("LYS",),
}


def _copyfile_if_needed(src, dst):
    """Copy `src` to `dst` unless they resolve to the same file."""
    try:
        if os.path.samefile(src, dst):
            return
    except FileNotFoundError:
        # Destination does not exist yet, fall back to regular copy.
        pass
    shutil.copyfile(src, dst)


def _normalize_membrane_system(membrane_system):
    """Validate and normalize membrane builder options."""
    if membrane_system is None:
        return None
    if not isinstance(membrane_system, Mapping):
        raise TypeError("membrane_system must be a mapping of membrane builder options.")

    lipid_type = str(membrane_system.get("lipid_type", "POPC")).strip().upper()
    if lipid_type not in _SUPPORTED_MEMBRANE_LIPIDS:
        supported = ", ".join(sorted(_SUPPORTED_MEMBRANE_LIPIDS))
        raise ValueError(
            f"Unsupported lipid_type {lipid_type!r}. Supported phase-1 membrane lipids: {supported}."
        )

    orientation_mode = str(membrane_system.get("orientation_mode", "as_is")).strip().lower()
    if orientation_mode not in _SUPPORTED_MEMBRANE_ORIENTATION_MODES:
        supported = ", ".join(sorted(_SUPPORTED_MEMBRANE_ORIENTATION_MODES))
        raise ValueError(
            f"Unsupported membrane orientation_mode {orientation_mode!r}. Supported values: {supported}."
        )

    minimum_padding_nm = float(membrane_system.get("minimum_padding_nm", 1.0))
    if minimum_padding_nm <= 0.0:
        raise ValueError("membrane_system minimum_padding_nm must be > 0.")

    membrane_center_z_nm = float(membrane_system.get("membrane_center_z_nm", 0.0))
    ionic_strength_molar = float(membrane_system.get("ionic_strength_molar", 0.15))
    if ionic_strength_molar < 0.0:
        raise ValueError("membrane_system ionic_strength_molar must be >= 0.")
    build_retries = int(membrane_system.get("build_retries", 1))
    if build_retries < 1:
        raise ValueError("membrane_system build_retries must be >= 1.")

    positive_ion = str(membrane_system.get("positive_ion", "Na+")).strip()
    negative_ion = str(membrane_system.get("negative_ion", "Cl-")).strip()
    if not positive_ion or not negative_ion:
        raise ValueError("membrane_system positive_ion and negative_ion must be non-empty.")

    exclude_chain_ids_raw = membrane_system.get("exclude_chain_ids", ())
    if exclude_chain_ids_raw is None:
        exclude_chain_ids = ()
    elif isinstance(exclude_chain_ids_raw, str):
        exclude_chain_ids = (exclude_chain_ids_raw.strip(),) if exclude_chain_ids_raw.strip() else ()
    else:
        try:
            exclude_chain_ids = tuple(
                chain_id.strip()
                for chain_id in exclude_chain_ids_raw
                if str(chain_id).strip()
            )
        except TypeError as exc:
            raise TypeError("membrane_system exclude_chain_ids must be a string or an iterable of chain ids.") from exc

    normalized = {
        "lipid_type": lipid_type,
        "orientation_mode": orientation_mode,
        "minimum_padding_nm": minimum_padding_nm,
        "membrane_center_z_nm": membrane_center_z_nm,
        "ionic_strength_molar": ionic_strength_molar,
        "build_retries": build_retries,
        "positive_ion": positive_ion,
        "negative_ion": negative_ion,
        "neutralize": bool(membrane_system.get("neutralize", True)),
        "exclude_chain_ids": exclude_chain_ids,
    }
    normalized.update(_SUPPORTED_MEMBRANE_LIPIDS[lipid_type])
    return normalized


def _run_command(command, command_log=None):
    """Execute `command` via `os.system` while optionally recording the call."""
    entry = None
    if command_log is not None:
        entry = {"command": command.rstrip(), "returncode": None}
        command_log.append(entry)
    ret = os.system(command)
    if entry is not None:
        entry["returncode"] = ret
    return ret


def _normalize_residue_id(value):
    """Return a residue identifier preserving integers when possible."""
    if value is None:
        raise ValueError("Residue identifier cannot be None.")
    text = str(value).strip()
    if not text:
        raise ValueError("Residue identifier cannot be empty.")
    try:
        return int(text)
    except ValueError:
        return text


def _normalize_optional_residue_name(value):
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _mcpb_structure_residue_candidates(residue_name, structure_residue_name=None):
    candidates = []
    if structure_residue_name:
        candidates.append(str(structure_residue_name).strip().upper())
    residue_name = str(residue_name).strip().upper()
    if residue_name:
        candidates.append(residue_name)
        candidates.extend(_MCPB_STRUCTURE_RESIDUE_ALIASES.get(residue_name, ()))
    return _ordered_unique([candidate for candidate in candidates if candidate])


def _normalize_mcpb_atom_spec(atom_spec, label="atom"):
    """Normalize an MCPB atom specification into a canonical mapping."""
    role = None
    structure_residue_name = None
    if isinstance(atom_spec, Mapping):
        chain_id = atom_spec.get("chain_id", atom_spec.get("chain"))
        residue_id = atom_spec.get("residue_id", atom_spec.get("resid", atom_spec.get("residue")))
        residue_name = atom_spec.get("residue_name", atom_spec.get("resname"))
        structure_residue_name = _normalize_optional_residue_name(
            atom_spec.get(
                "structure_residue_name",
                atom_spec.get(
                    "structure_resname",
                    atom_spec.get("input_residue_name", atom_spec.get("input_resname")),
                ),
            )
        )
        atom_name = atom_spec.get("atom_name", atom_spec.get("atom"))
        role = atom_spec.get("role")
    elif isinstance(atom_spec, (tuple, list)) and len(atom_spec) == 4:
        chain_id, residue_id, residue_name, atom_name = atom_spec
    else:
        raise TypeError(
            f"{label} must be a mapping or a 4-item sequence "
            "(chain_id, residue_id, residue_name, atom_name)."
        )

    if chain_id is None:
        raise ValueError(f"{label} is missing 'chain_id'.")
    if residue_name is None:
        raise ValueError(f"{label} is missing 'residue_name'.")
    if atom_name is None:
        raise ValueError(f"{label} is missing 'atom_name'.")

    chain_text = str(chain_id).strip()
    residue_name_text = str(residue_name).strip().upper()
    atom_name_text = str(atom_name).strip().upper()
    if not chain_text:
        raise ValueError(f"{label} chain_id cannot be empty.")
    if not residue_name_text:
        raise ValueError(f"{label} residue_name cannot be empty.")
    if not atom_name_text:
        raise ValueError(f"{label} atom_name cannot be empty.")

    normalized = {
        "chain_id": chain_text,
        "residue_id": _normalize_residue_id(residue_id),
        "residue_name": residue_name_text,
        "atom_name": atom_name_text,
        "lookup_residue_names": _mcpb_structure_residue_candidates(
            residue_name_text,
            structure_residue_name=structure_residue_name,
        ),
    }
    if structure_residue_name is not None:
        normalized["structure_residue_name"] = structure_residue_name
    if role is not None:
        role_text = str(role).strip()
        if role_text:
            normalized["role"] = role_text
    return normalized


def _normalize_mcpb_site(site_spec, index):
    """Normalize a single MCPB site definition."""
    if not isinstance(site_spec, Mapping):
        raise TypeError("Each MCPB site definition must be a mapping.")

    site_id = site_spec.get("site_id", site_spec.get("id"))
    if site_id is None:
        site_id = f"site_{index}"
    site_id = str(site_id).strip()
    if not site_id:
        raise ValueError("MCPB site_id cannot be empty.")

    metal_spec = site_spec.get("metal", site_spec.get("metal_atom"))
    if metal_spec is None:
        raise ValueError(f"MCPB site '{site_id}' is missing the 'metal' atom specification.")

    coordinating_atoms = (
        site_spec.get("coordinating_atoms")
        or site_spec.get("ligating_atoms")
        or site_spec.get("coordinators")
        or site_spec.get("atoms")
    )
    if coordinating_atoms is None:
        raise ValueError(
            f"MCPB site '{site_id}' is missing 'coordinating_atoms'."
        )
    if not isinstance(coordinating_atoms, (list, tuple)):
        raise TypeError(f"MCPB site '{site_id}' coordinating_atoms must be a list or tuple.")
    if not coordinating_atoms:
        raise ValueError(f"MCPB site '{site_id}' must define at least one coordinating atom.")

    cut_off = site_spec.get("cut_off", 2.8)
    try:
        cut_off = float(cut_off)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MCPB site '{site_id}' cut_off must be numeric.") from exc

    normalized_site = {
        "site_id": site_id,
        "group_name": str(site_spec.get("group_name", site_id)).strip() or site_id,
        "metal": _normalize_mcpb_atom_spec(metal_spec, label=f"MCPB site '{site_id}' metal"),
        "coordinating_atoms": [
            _normalize_mcpb_atom_spec(atom_spec, label=f"MCPB site '{site_id}' coordinating atom {atom_index + 1}")
            for atom_index, atom_spec in enumerate(coordinating_atoms)
        ],
        "cut_off": cut_off,
    }
    if "notes" in site_spec and site_spec["notes"] is not None:
        normalized_site["notes"] = str(site_spec["notes"])
    return normalized_site


def _normalize_mcpb_config(mcpb_config):
    """Normalize the public MCPB configuration into a stable internal structure."""
    if not mcpb_config:
        return None

    if isinstance(mcpb_config, Mapping) and "sites" in mcpb_config:
        sites = mcpb_config["sites"]
        top_level_notes = mcpb_config.get("notes")
    elif isinstance(mcpb_config, Mapping):
        sites = [mcpb_config]
        top_level_notes = None
    elif isinstance(mcpb_config, (list, tuple)):
        sites = list(mcpb_config)
        top_level_notes = None
    else:
        raise TypeError(
            "mcpb_config must be a site mapping, a list of site mappings, or a mapping with a 'sites' entry."
        )

    if not sites:
        raise ValueError("mcpb_config must contain at least one site.")

    normalized = {
        "sites": [_normalize_mcpb_site(site_spec, index + 1) for index, site_spec in enumerate(sites)]
    }
    if top_level_notes is not None:
        normalized["notes"] = str(top_level_notes)
    return normalized


def _iter_topology_atoms(topology):
    for chain in topology.chains():
        for residue in chain.residues():
            residue_atoms = list(residue.atoms())
            normalized_residue_id = _normalize_residue_id(residue.id)
            for atom in residue_atoms:
                yield {
                    "chain_id": chain.id,
                    "residue_id": normalized_residue_id,
                    "residue_name": residue.name.upper(),
                    "atom_name": atom.name.strip().upper(),
                    "atom_index": atom.index,
                    "element": (
                        getattr(getattr(atom, "element", None), "symbol", None)
                        or getattr(getattr(atom, "element", None), "name", None)
                        or atom.name[:1]
                    ).upper(),
                    "residue_atom_count": len(residue_atoms),
                }


def _position_to_angstrom(position):
    if hasattr(position, "value_in_unit"):
        values = position.value_in_unit(nanometer)
    else:
        values = position
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3D position, got shape {arr.shape}.")
    return arr * 10.0


def _resolve_mcpb_config(topology, positions, mcpb_config):
    """Validate and enrich MCPB site definitions against a topology."""
    normalized = _normalize_mcpb_config(mcpb_config)
    if normalized is None:
        return None

    atom_lookup = {}
    residue_lookup = defaultdict(list)
    for atom_record in _iter_topology_atoms(topology):
        key = (
            atom_record["chain_id"],
            atom_record["residue_id"],
            atom_record["residue_name"],
            atom_record["atom_name"],
        )
        atom_lookup[key] = atom_record
        residue_key = (
            atom_record["chain_id"],
            atom_record["residue_id"],
            atom_record["residue_name"],
        )
        residue_lookup[residue_key].append(atom_record)

    positions_angstrom = None
    if positions is not None:
        try:
            positions_angstrom = [_position_to_angstrom(position) for position in positions]
        except Exception:
            positions_angstrom = None

    resolved_sites = []
    for site in normalized["sites"]:
        resolved_site = dict(site)
        metal_record = _resolve_mcpb_atom_record(
            atom_lookup,
            dict(site["metal"]),
            site["site_id"],
            "metal atom",
        )
        metal_record["is_embedded_metal"] = metal_record["residue_atom_count"] > 1
        if positions_angstrom is not None:
            metal_record["position_angstrom"] = positions_angstrom[metal_record["atom_index"]].tolist()
        resolved_site["metal"] = metal_record

        resolved_coordinating_atoms = []
        site_residues = {
            (
                metal_record["chain_id"],
                metal_record["residue_id"],
                metal_record["residue_name"],
            )
        }
        for atom_spec in site["coordinating_atoms"]:
            atom_record = _resolve_mcpb_atom_record(
                atom_lookup,
                dict(atom_spec),
                site["site_id"],
                "coordinating atom",
            )
            if positions_angstrom is not None:
                atom_record["position_angstrom"] = positions_angstrom[atom_record["atom_index"]].tolist()
                metal_pos = positions_angstrom[metal_record["atom_index"]]
                atom_pos = positions_angstrom[atom_record["atom_index"]]
                atom_record["distance_to_metal_angstrom"] = float(np.linalg.norm(atom_pos - metal_pos))
            resolved_coordinating_atoms.append(atom_record)
            site_residues.add(
                (
                    atom_record["chain_id"],
                    atom_record["residue_id"],
                    atom_record["residue_name"],
                )
            )

        resolved_site["coordinating_atoms"] = resolved_coordinating_atoms
        resolved_site["embedded_metal"] = bool(metal_record["is_embedded_metal"])
        resolved_site["legacy_compatible"] = not resolved_site["embedded_metal"]
        resolved_site["residues"] = [
            {
                "chain_id": residue_key[0],
                "residue_id": residue_key[1],
                "residue_name": residue_key[2],
                "atom_count": len(residue_lookup[residue_key]),
            }
            for residue_key in sorted(site_residues, key=lambda item: (item[0], str(item[1]), item[2]))
        ]
        resolved_sites.append(resolved_site)

    normalized["sites"] = resolved_sites
    normalized["contains_embedded_metal"] = any(site["embedded_metal"] for site in resolved_sites)
    normalized["legacy_compatible"] = all(site["legacy_compatible"] for site in resolved_sites)
    return normalized


def _legacy_metal_ligand_from_mcpb_config(resolved_mcpb_config):
    """Best-effort conversion of validated MCPB sites to the legacy metal_ligand API."""
    if not resolved_mcpb_config or not resolved_mcpb_config.get("legacy_compatible", False):
        return None

    legacy = defaultdict(list)
    for site in resolved_mcpb_config["sites"]:
        metal_residue_name = site["metal"]["residue_name"]
        for atom_record in site["coordinating_atoms"]:
            residue_name = atom_record["residue_name"]
            if atom_record.get("is_protein_residue", residue_name in aa3):
                continue
            if residue_name == metal_residue_name:
                continue
            if metal_residue_name not in legacy[residue_name]:
                legacy[residue_name].append(metal_residue_name)
    return dict(legacy)


def _ordered_unique(values):
    ordered = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _resolve_residue_parameter_file(residue_name, registered_files, par_folder, parameters_folder, extension):
    """Return the primary parameter file for `residue_name` if it exists."""
    residue_name = str(residue_name).upper()
    for path in registered_files.get(residue_name, []):
        if path and os.path.exists(path):
            return path

    if residue_name in par_folder:
        candidate = os.path.join(par_folder[residue_name], f"{residue_name}{extension}")
        if os.path.exists(candidate):
            return candidate

    candidate = os.path.join(parameters_folder, f"{residue_name}{extension}")
    if os.path.exists(candidate):
        return candidate
    return None


def _build_site_driven_mcpb_inputs(
    resolved_mcpb_config,
    par_folder,
    residue_mol2_files,
    residue_frcmod_files,
    parameters_folder,
    pdb_name,
):
    """Build MCPB input metadata from validated atom-level MCPB site definitions."""
    if not resolved_mcpb_config:
        return None

    sites = resolved_mcpb_config.get("sites", [])
    if not sites:
        raise ValueError("resolved_mcpb_config does not contain any sites.")

    ion_ids = []
    bonded_pairs = []
    bonded_pair_seen = set()
    selected_residue_names = []
    non_protein_site_residues = []
    staged_residue_names = []
    ion_mol2_files = []

    cutoffs = []
    for site in sites:
        cutoffs.append(float(site.get("cut_off", 2.8)))

        metal_record = site["metal"]
        metal_residue_name = metal_record["residue_name"].upper()
        ion_ids.append(metal_record["atom_index"] + 1)
        if metal_residue_name not in aa3:
            staged_residue_names.append(metal_residue_name)

        if metal_residue_name not in aa3 and metal_record.get("is_embedded_metal", False):
            selected_residue_names.append(metal_residue_name)
            non_protein_site_residues.append(metal_residue_name)
        elif metal_residue_name not in aa3:
            metal_mol2 = _resolve_residue_parameter_file(
                metal_residue_name,
                residue_mol2_files,
                par_folder,
                parameters_folder,
                ".mol2",
            )
            if metal_mol2 is not None:
                ion_mol2_files.append(f"{metal_residue_name}.mol2")

        for atom_record in site["coordinating_atoms"]:
            residue_name = atom_record["residue_name"].upper()
            pair = (metal_record["atom_index"] + 1, atom_record["atom_index"] + 1)
            if pair not in bonded_pair_seen:
                bonded_pair_seen.add(pair)
                bonded_pairs.append(pair)
            if atom_record.get("is_protein_residue", residue_name in aa3):
                continue
            selected_residue_names.append(residue_name)
            non_protein_site_residues.append(residue_name)
            staged_residue_names.append(residue_name)

    selected_residue_names = _ordered_unique(selected_residue_names)
    non_protein_site_residues = _ordered_unique(non_protein_site_residues)
    staged_residue_names = _ordered_unique(staged_residue_names)
    ion_ids = _ordered_unique(ion_ids)
    ion_mol2_files = _ordered_unique(ion_mol2_files)

    missing_mol2 = []
    naa_mol2_files = []
    for residue_name in non_protein_site_residues:
        mol2_path = _resolve_residue_parameter_file(
            residue_name,
            residue_mol2_files,
            par_folder,
            parameters_folder,
            ".mol2",
        )
        if mol2_path is None:
            missing_mol2.append(residue_name)
            continue
        naa_mol2_files.append(f"{residue_name}.mol2")

    if missing_mol2:
        raise ValueError(
            "MCPB site-driven setup requires MOL2 files for the selected non-protein site residues. "
            f"Missing mol2 files for: {', '.join(sorted(missing_mol2))}."
        )

    missing_frcmod = []
    frcmod_files = []
    for residue_name in non_protein_site_residues:
        frcmod_path = _resolve_residue_parameter_file(
            residue_name,
            residue_frcmod_files,
            par_folder,
            parameters_folder,
            ".frcmod",
        )
        if frcmod_path is None:
            missing_frcmod.append(residue_name)
            continue
        frcmod_files.append(f"{residue_name}.frcmod")

    if missing_frcmod:
        raise ValueError(
            "MCPB site-driven setup requires frcmod files for the selected non-protein site residues. "
            f"Missing frcmod files for: {', '.join(sorted(missing_frcmod))}."
        )

    unique_cutoffs = sorted({round(cutoff, 6) for cutoff in cutoffs})
    if len(unique_cutoffs) > 1:
        warnings.warn(
            "Multiple MCPB site cutoffs were provided. Using the largest value for the combined MCPB input file.",
            RuntimeWarning,
        )

    return {
        "group_name": pdb_name,
        "cut_off": max(cutoffs) if cutoffs else 2.8,
        "ion_ids": ion_ids,
        "ion_mol2files": ion_mol2_files,
        "naa_mol2files": naa_mol2_files,
        "frcmod_files": frcmod_files,
        "add_bonded_pairs": bonded_pairs,
        "site_residue_names": non_protein_site_residues,
        "staged_residue_names": staged_residue_names,
        "selected_residue_names": selected_residue_names,
    }


def _collect_mcpb_nonprotein_residue_selection(resolved_mcpb_config):
    """Collect exact non-protein residue instances selected by mcpb_config."""
    selection = {
        "keys": [],
        "names": [],
        "entries": [],
        "embedded_residue_names": [],
        "duplicate_names": {},
    }
    if not resolved_mcpb_config:
        return selection

    entry_by_key = {}
    ordered_keys = []
    for site in resolved_mcpb_config.get("sites", []):
        for atom_record in [site["metal"], *site["coordinating_atoms"]]:
            residue_name = atom_record["residue_name"].upper()
            if atom_record.get("is_protein_residue", residue_name in aa3):
                continue
            residue_key = (
                atom_record["chain_id"],
                atom_record["residue_id"],
                residue_name,
            )
            entry = entry_by_key.get(residue_key)
            if entry is None:
                entry = {
                    "chain_id": atom_record["chain_id"],
                    "residue_id": atom_record["residue_id"],
                    "residue_name": residue_name,
                    "contains_embedded_metal": False,
                }
                entry_by_key[residue_key] = entry
                ordered_keys.append(residue_key)
            if atom_record.get("is_embedded_metal", False):
                entry["contains_embedded_metal"] = True

    ordered_entries = [entry_by_key[key] for key in ordered_keys]
    duplicate_names = defaultdict(list)
    embedded_residue_names = []
    for entry in ordered_entries:
        duplicate_names[entry["residue_name"]].append(entry)
        if entry["contains_embedded_metal"] and entry["residue_name"] not in embedded_residue_names:
            embedded_residue_names.append(entry["residue_name"])

    selection["keys"] = ordered_keys
    selection["names"] = _ordered_unique([entry["residue_name"] for entry in ordered_entries])
    selection["entries"] = ordered_entries
    selection["embedded_residue_names"] = embedded_residue_names
    selection["duplicate_names"] = {
        residue_name: entries
        for residue_name, entries in duplicate_names.items()
        if len(entries) > 1
    }
    return selection


def _normalize_required_int(value, label):
    if value is None:
        raise ValueError(f"{label} is required.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _normalize_mcpb_qm_settings(site_spec, site_id):
    qm_spec = site_spec.get("qm")
    if not isinstance(qm_spec, Mapping):
        raise ValueError(f"MCPB site '{site_id}' must define a 'qm' mapping.")

    software_version = str(qm_spec.get("software_version", "g09")).strip()
    if not software_version:
        software_version = "g09"

    small_model_charge = _normalize_required_int(
        qm_spec.get("small_model_charge", qm_spec.get("smmodel_chg")),
        f"MCPB site '{site_id}' qm.small_model_charge",
    )
    small_model_spin = _normalize_required_int(
        qm_spec.get("small_model_spin", qm_spec.get("smmodel_spin")),
        f"MCPB site '{site_id}' qm.small_model_spin",
    )
    large_model_charge = _normalize_required_int(
        qm_spec.get("large_model_charge", qm_spec.get("lgmodel_chg")),
        f"MCPB site '{site_id}' qm.large_model_charge",
    )
    large_model_spin = _normalize_required_int(
        qm_spec.get("large_model_spin", qm_spec.get("lgmodel_spin")),
        f"MCPB site '{site_id}' qm.large_model_spin",
    )

    if small_model_spin <= 0:
        raise ValueError(f"MCPB site '{site_id}' qm.small_model_spin must be >= 1.")
    if large_model_spin <= 0:
        raise ValueError(f"MCPB site '{site_id}' qm.large_model_spin must be >= 1.")

    return {
        "software_version": software_version,
        "small_model_charge": small_model_charge,
        "small_model_spin": small_model_spin,
        "large_model_charge": large_model_charge,
        "large_model_spin": large_model_spin,
    }


def _normalize_mcpb_fragment_specs(site_spec, site_id):
    fragment_specs = (
        site_spec.get("fragments")
        or site_spec.get("nonstandard_fragments")
        or site_spec.get("ligand_fragments")
        or []
    )
    if isinstance(fragment_specs, Mapping):
        fragment_specs = [fragment_specs]
    if not isinstance(fragment_specs, (list, tuple)):
        raise TypeError(f"MCPB site '{site_id}' fragments must be a list of mappings.")

    normalized_fragments = []
    for index, fragment_spec in enumerate(fragment_specs, start=1):
        if not isinstance(fragment_spec, Mapping):
            raise TypeError(f"MCPB site '{site_id}' fragment {index} must be a mapping.")
        chain_id = fragment_spec.get("chain_id", fragment_spec.get("chain"))
        residue_id = fragment_spec.get("residue_id", fragment_spec.get("resid", fragment_spec.get("residue")))
        residue_name = fragment_spec.get("residue_name", fragment_spec.get("resname"))
        structure_residue_name = _normalize_optional_residue_name(
            fragment_spec.get(
                "structure_residue_name",
                fragment_spec.get(
                    "structure_resname",
                    fragment_spec.get("input_residue_name", fragment_spec.get("input_resname")),
                ),
            )
        )
        if chain_id is None:
            raise ValueError(f"MCPB site '{site_id}' fragment {index} is missing 'chain_id'.")
        if residue_name is None:
            raise ValueError(f"MCPB site '{site_id}' fragment {index} is missing 'residue_name'.")
        chain_id = str(chain_id).strip()
        residue_name = str(residue_name).strip().upper()
        if not chain_id:
            raise ValueError(f"MCPB site '{site_id}' fragment {index} chain_id cannot be empty.")
        if not residue_name:
            raise ValueError(f"MCPB site '{site_id}' fragment {index} residue_name cannot be empty.")
        net_charge = _normalize_required_int(
            fragment_spec.get("net_charge", fragment_spec.get("charge")),
            f"MCPB site '{site_id}' fragment {index} net_charge",
        )
        exclude_atoms = fragment_spec.get("exclude_atoms", fragment_spec.get("omit_atoms", []))
        if isinstance(exclude_atoms, str):
            exclude_atoms = [exclude_atoms]
        elif exclude_atoms is None:
            exclude_atoms = []
        elif not isinstance(exclude_atoms, (list, tuple, set)):
            raise TypeError(
                f"MCPB site '{site_id}' fragment {index} exclude_atoms must be a string or a list of strings."
            )
        exclude_atoms = [str(atom_name).strip().upper() for atom_name in exclude_atoms if str(atom_name).strip()]

        output_name = str(
            fragment_spec.get("output_name", fragment_spec.get("name", residue_name))
        ).strip().upper()
        if not output_name:
            raise ValueError(f"MCPB site '{site_id}' fragment {index} output_name cannot be empty.")

        normalized = {
            "chain_id": chain_id,
            "residue_id": _normalize_residue_id(residue_id),
            "residue_name": residue_name,
        }
        if structure_residue_name is not None:
            normalized["structure_residue_name"] = structure_residue_name
        normalized["net_charge"] = net_charge
        normalized["exclude_atoms"] = exclude_atoms
        normalized["output_name"] = output_name
        normalized_fragments.append(normalized)

    return normalized_fragments


def _normalize_preparable_mcpb_site(mcpb_site):
    if not mcpb_site:
        raise ValueError("mcpb_site is required.")

    if isinstance(mcpb_site, Mapping) and "sites" in mcpb_site:
        sites = mcpb_site.get("sites") or []
        if len(sites) != 1:
            raise ValueError("prepareMCPBSite currently supports exactly one MCPB site definition.")
        site_spec = sites[0]
    elif isinstance(mcpb_site, Mapping):
        site_spec = mcpb_site
    else:
        raise TypeError("mcpb_site must be a site mapping or a mapping containing exactly one site.")

    normalized_site = _normalize_mcpb_site(site_spec, 1)
    site_id = normalized_site["site_id"]

    metal_spec = dict(normalized_site["metal"])
    metal_spec["formal_charge"] = _normalize_required_int(
        site_spec.get("metal", {}).get("formal_charge")
        if isinstance(site_spec.get("metal"), Mapping)
        else None,
        f"MCPB site '{site_id}' metal formal_charge",
    )
    metal_output_residue_name = str(
        site_spec.get("metal", {}).get("output_residue_name", site_spec.get("metal", {}).get("output_resname", metal_spec["atom_name"]))
        if isinstance(site_spec.get("metal"), Mapping)
        else metal_spec["atom_name"]
    ).strip().upper()
    metal_output_atom_name = str(
        site_spec.get("metal", {}).get("output_atom_name", metal_spec["atom_name"])
        if isinstance(site_spec.get("metal"), Mapping)
        else metal_spec["atom_name"]
    ).strip().upper()
    if not metal_output_residue_name:
        metal_output_residue_name = metal_spec["atom_name"]
    if not metal_output_atom_name:
        metal_output_atom_name = metal_spec["atom_name"]
    metal_spec["output_residue_name"] = metal_output_residue_name
    metal_spec["output_atom_name"] = metal_output_atom_name

    normalized_site["metal"] = metal_spec
    normalized_site["qm"] = _normalize_mcpb_qm_settings(site_spec, site_id)
    normalized_site["fragments"] = _normalize_mcpb_fragment_specs(site_spec, site_id)
    return normalized_site


def _find_residue_in_topology(topology, chain_id, residue_id, residue_name, structure_residue_name=None):
    matches = []
    normalized_residue_id = _normalize_residue_id(residue_id)
    lookup_residue_names = set(
        _mcpb_structure_residue_candidates(residue_name, structure_residue_name=structure_residue_name)
    )
    for chain in topology.chains():
        if chain.id != chain_id:
            continue
        for residue in chain.residues():
            if _normalize_residue_id(residue.id) != normalized_residue_id:
                continue
            if residue.name.upper() not in lookup_residue_names:
                continue
            matches.append(residue)
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one residue "
            f"{residue_name} {chain_id} {normalized_residue_id} "
            f"(lookup names: {sorted(lookup_residue_names)}), found {len(matches)}."
        )
    return matches[0]


def _find_atom_in_topology(topology, atom_spec):
    residue = _find_residue_in_topology(
        topology,
        atom_spec["chain_id"],
        atom_spec["residue_id"],
        atom_spec["residue_name"],
        structure_residue_name=atom_spec.get("structure_residue_name"),
    )
    matches = [atom for atom in residue.atoms() if atom.name.strip().upper() == atom_spec["atom_name"]]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one atom {atom_spec['atom_name']} in "
            f"{atom_spec['residue_name']} {atom_spec['chain_id']} {atom_spec['residue_id']}, found {len(matches)}."
        )
    return matches[0]


def _resolve_mcpb_atom_record(atom_lookup, atom_spec, site_id, atom_label):
    matches = {}
    for residue_name in atom_spec.get("lookup_residue_names", [atom_spec["residue_name"]]):
        key = (
            atom_spec["chain_id"],
            atom_spec["residue_id"],
            residue_name,
            atom_spec["atom_name"],
        )
        atom_record = atom_lookup.get(key)
        if atom_record is not None:
            matches[atom_record["atom_index"]] = atom_record

    if len(matches) != 1:
        raise ValueError(
            f"MCPB site '{site_id}' {atom_label} {atom_spec} was not found in the structure. "
            f"Lookup residue names: {atom_spec.get('lookup_residue_names', [atom_spec['residue_name']])}."
        )

    matched_record = dict(next(iter(matches.values())))
    matched_record.update(atom_spec)
    matched_record["structure_residue_name"] = next(iter(matches.values()))["residue_name"]
    matched_record["requested_residue_name"] = atom_spec["residue_name"]
    matched_record["is_protein_residue"] = matched_record["structure_residue_name"].upper() in aa3
    return matched_record


def _extract_residue_subsystem_filtered(
    modeller,
    residue,
    exclude_atoms=None,
    residue_name=None,
    chain_id=None,
    residue_id=None,
):
    _ensure_openmm("filtered residue extraction")
    exclude_set = {str(atom_name).strip().upper() for atom_name in (exclude_atoms or []) if str(atom_name).strip()}

    top = Topology()
    chain = top.addChain(chain_id if chain_id is not None else residue.chain.id)
    new_residue = top.addResidue(
        residue_name if residue_name is not None else residue.name,
        chain,
        id=str(residue_id if residue_id is not None else residue.id),
    )

    atom_map = {}
    positions = []
    modeller_positions = modeller.getPositions()
    for atom in modeller.topology.atoms():
        if atom.residue != residue:
            continue
        if atom.name.strip().upper() in exclude_set:
            continue
        new_atom = top.addAtom(atom.name, atom.element, new_residue)
        atom_map[atom.index] = new_atom
        pos = modeller_positions[atom.index]
        if hasattr(pos, "value_in_unit"):
            xyz = pos.value_in_unit(nanometer)
            pos = Vec3(xyz[0], xyz[1], xyz[2])
        positions.append(pos)

    if not positions:
        raise ValueError(f"Residue '{residue.name}' has no atoms left after filtering.")

    for atom1, atom2 in modeller.topology.bonds():
        if atom1.index in atom_map and atom2.index in atom_map:
            top.addBond(atom_map[atom1.index], atom_map[atom2.index])

    return top, Quantity(positions, nanometer)


def _write_filtered_residue_pdb(
    modeller,
    residue,
    output_path,
    exclude_atoms=None,
    residue_name=None,
    chain_id=None,
    residue_id=None,
):
    top, positions = _extract_residue_subsystem_filtered(
        modeller,
        residue,
        exclude_atoms=exclude_atoms,
        residue_name=residue_name,
        chain_id=chain_id,
        residue_id=residue_id,
    )
    with open(output_path, "w") as handle:
        PDBFile.writeFile(top, positions, handle)
    return top, positions


def _write_single_atom_pdb(
    atom_record,
    output_path,
    residue_name,
    atom_name,
    residue_id=1,
    chain_id="A",
):
    _ensure_openmm("single-atom MCPB fragment writing")
    top = Topology()
    chain = top.addChain(str(chain_id))
    residue = top.addResidue(str(residue_name), chain, id=str(residue_id))
    element_symbol = atom_record.get("element", atom_name).upper()
    element_obj = Element.getBySymbol(element_symbol)
    top.addAtom(str(atom_name), element_obj, residue)
    position_angstrom = atom_record.get("position_angstrom")
    if position_angstrom is None:
        raise ValueError("Atom position is required to write a single-atom PDB fragment.")
    positions = Quantity([Vec3(*(np.asarray(position_angstrom, dtype=float) / 10.0))], nanometer)
    with open(output_path, "w") as handle:
        PDBFile.writeFile(top, positions, handle)


def _build_split_mcpb_modeller(modeller, resolved_site, metal_output_residue_name, metal_output_atom_name):
    _ensure_openmm("MCPB site splitting")
    split_modeller = Modeller(modeller.topology, modeller.positions)
    metal_atom = _find_atom_in_topology(split_modeller.topology, resolved_site["metal"])
    metal_position = split_modeller.positions[metal_atom.index]
    metal_chain_id = metal_atom.residue.chain.id
    metal_element = metal_atom.element

    split_modeller.delete([metal_atom])

    max_residue_id = 0
    existing_chain_ids = set()
    for chain in split_modeller.topology.chains():
        existing_chain_ids.add(chain.id)
        for residue in chain.residues():
            try:
                max_residue_id = max(max_residue_id, int(str(residue.id).strip()))
            except ValueError:
                continue

    chain_candidates = [metal_chain_id]
    chain_candidates.extend(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    chain_candidates.extend(list("0123456789"))
    chain_candidates.extend(
        [f"M{index}" for index in range(1, len(existing_chain_ids) + 10)]
    )

    new_chain_id = None
    for candidate in chain_candidates:
        if candidate not in existing_chain_ids:
            new_chain_id = candidate
            break
    if new_chain_id is None:
        raise ValueError("Could not assign a unique chain identifier for the split MCPB metal residue.")

    chain_obj = split_modeller.topology.addChain(new_chain_id)

    new_residue_id = str(max_residue_id + 1 if max_residue_id > 0 else 1)
    metal_residue = split_modeller.topology.addResidue(metal_output_residue_name, chain_obj, id=new_residue_id)
    split_modeller.topology.addAtom(metal_output_atom_name, metal_element, metal_residue)
    split_modeller.positions.append(metal_position)

    metal_spec = {
        "chain_id": new_chain_id,
        "residue_id": _normalize_residue_id(new_residue_id),
        "residue_name": metal_output_residue_name.upper(),
        "atom_name": metal_output_atom_name.upper(),
    }
    return split_modeller, metal_spec


def _pdb_atom_serials_by_topology_index(pdb_path, expected_atom_count=None):
    """Return the written PDB atom serial number for each 0-based topology atom index."""
    atom_serials = []
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            serial_text = line[6:11].strip()
            if not serial_text:
                raise ValueError(f"Missing atom serial number in PDB record: {line.rstrip()}")
            atom_serials.append(int(serial_text))

    if expected_atom_count is not None and len(atom_serials) != expected_atom_count:
        raise ValueError(
            f"PDB file {pdb_path} contains {len(atom_serials)} atom records, "
            f"but {expected_atom_count} topology atoms were expected."
        )

    return {atom_index: serial for atom_index, serial in enumerate(atom_serials)}


def _renumber_pdb_residues_globally(pdb_path):
    """Rewrite a PDB so residue sequence numbers are unique across all chains."""
    residue_serial_by_key = {}
    next_residue_serial = 1
    current_residue_key = None
    rewritten_lines = []

    with open(pdb_path) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                residue_key = (line[21], line[22:26], line[26], line[17:20])
                if residue_key not in residue_serial_by_key:
                    residue_serial_by_key[residue_key] = next_residue_serial
                    next_residue_serial += 1
                current_residue_key = residue_key
                new_residue_serial = residue_serial_by_key[residue_key]
                line = line[:22] + f"{new_residue_serial:4d}" + line[26:]
            elif line.startswith("TER") and current_residue_key is not None:
                new_residue_serial = residue_serial_by_key[current_residue_key]
                line = line[:22] + f"{new_residue_serial:4d}" + line[26:]
            rewritten_lines.append(line)

    with open(pdb_path, "w") as handle:
        handle.writelines(rewritten_lines)


class openmm_md:

    def __init__(self, input_pdb):
        _ensure_openmm("openmm_md")

        # Set variables
        self.input_pdb = input_pdb
        self.pdb_name = input_pdb.split('/')[-1].replace('.pdb', '')
        self.pdb = pdb = PDBFile(self.input_pdb)
        self.modeller = Modeller(pdb.topology, pdb.positions)
        self.positions = np.array([c.value_in_unit(nanometer) for c in self.modeller.positions])
        self.command_log = []
        self.membrane_system = None
        self._membrane_built = False

    def _refresh_positions_cache(self):
        self.positions = np.array([c.value_in_unit(nanometer) for c in self.modeller.positions])

    def setUpFF(self, ff_name, membrane_system=None):

        membrane_config = _normalize_membrane_system(membrane_system)

        preset_ffs = {
            'amber14': ['amber14-all.xml', 'amber14/tip3pfb.xml'],
            'charmm36': ['charmm36.xml'],
        }

        if isinstance(ff_name, (list, tuple)):
            if len(ff_name) == 0:
                raise ValueError('Custom forcefield file list must not be empty.')
            self.ff_name = 'custom'
            self.ff_files = [str(ff_file) for ff_file in ff_name]
        else:
            if ff_name not in preset_ffs:
                available_ffs = sorted(preset_ffs)
                raise ValueError(f'{ff_name} not found in available forcefields: {available_ffs}')
            self.ff_name = ff_name
            if self.ff_name == 'amber14' and membrane_config:
                self.ff_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3pfb.xml', 'amber14/lipid17.xml']
            else:
                self.ff_files = preset_ffs[self.ff_name]
        self.forcefield = ForceField(*self.ff_files)
        self.membrane_system = membrane_config

    def setPeriodicBoundaryConditions(self, radius=1.5):
        min_xyz = np.min(self.positions, axis=0)
        max_xyz = np.max(self.positions, axis=0)
        vectors = [Vec3((max_xyz[0]-min_xyz[0])+radius, 0, 0),
                   Vec3(0, (max_xyz[1]-min_xyz[1])+radius, 0),
                   Vec3(0, 0, (max_xyz[2]-min_xyz[2])+radius)]
        self.modeller.topology.setPeriodicBoxVectors(vectors)

    def removeHydrogens(self, keep_ligands=True):
        hydrogens = [a for a in self.modeller.topology.atoms() if a.element.name == 'hydrogen']
        if keep_ligands:
            hydrogens = [a for a in hydrogens if a.residue.name in aa3]
        self.modeller.delete(hydrogens)
        self._refresh_positions_cache()

    def _extract_non_protein_residues(self, preserved_residue_names=None):
        """Temporarily remove non-protein residues and return cached residue data."""
        saved_residues = []
        non_protein = []
        protein_set = set(aa3) - set(ions)
        preserved_residue_names = {
            str(residue_name).strip().upper()
            for residue_name in (preserved_residue_names or [])
            if str(residue_name).strip()
        }

        # Precompute per-residue internal bonds for non-protein residues so we
        # can restore them after the temporary operation.
        internal_bonds = {}
        for bond in self.modeller.topology.bonds():
            if bond[0].residue == bond[1].residue:
                res = bond[0].residue
                internal_bonds.setdefault(res, []).append((bond[0].name, bond[1].name))

        for residue in self.modeller.topology.residues():
            residue_name = residue.name.upper()
            if residue_name in protein_set or residue_name in preserved_residue_names:
                continue
            atom_names = []
            atom_elements = []
            atom_positions = []
            for atom in residue.atoms():
                atom_names.append(atom.name)
                atom_elements.append(atom.element)
                atom_positions.append(self.modeller.positions[atom.index])
            saved_residues.append({
                "chain_id": residue.chain.id,
                "residue_id": str(residue.id),
                "residue_name": residue.name,
                "atom_names": atom_names,
                "atom_elements": atom_elements,
                "positions": atom_positions,
                "bonds": internal_bonds.get(residue, []),
            })
            non_protein.append(residue)

        return saved_residues, non_protein

    def _extract_chain_subsystem(self, excluded_chain_ids=None):
        """Temporarily remove complete chains and return a cached subsystem plus residues to delete."""
        excluded_chain_ids = {
            str(chain_id).strip()
            for chain_id in (excluded_chain_ids or [])
            if str(chain_id).strip()
        }
        if not excluded_chain_ids:
            return None, []

        topology = Topology()
        if self.modeller.topology.getPeriodicBoxVectors() is not None:
            topology.setPeriodicBoxVectors(self.modeller.topology.getPeriodicBoxVectors())

        atom_map = {}
        positions = []
        residues_to_delete = []

        for chain in self.modeller.topology.chains():
            if chain.id not in excluded_chain_ids:
                continue
            new_chain = topology.addChain(chain.id)
            for residue in chain.residues():
                residues_to_delete.append(residue)
                new_residue = topology.addResidue(residue.name, new_chain, id=str(residue.id))
                for atom in residue.atoms():
                    atom_map[atom] = topology.addAtom(atom.name, atom.element, new_residue)
                    positions.append(self.modeller.positions[atom.index])

        if not atom_map:
            return None, []

        for atom1, atom2 in self.modeller.topology.bonds():
            if atom1 in atom_map and atom2 in atom_map:
                topology.addBond(atom_map[atom1], atom_map[atom2])

        return (topology, positions), residues_to_delete

    def _reinsert_cached_residues(self, saved_residues):
        """Reinsert cached residues, preserving chain ids and internal bonds."""
        if not saved_residues:
            return

        # Group by chain id
        by_chain = {}
        for entry in saved_residues:
            by_chain.setdefault(entry["chain_id"], []).append(entry)

        # Existing chains in declared order.
        chain_objs = {c.id: c for c in self.modeller.topology.chains()}
        chain_order = [c.id for c in self.modeller.topology.chains()]

        # First, add residues to chains that already exist, in order.
        for cid in chain_order:
            if cid not in by_chain:
                continue
            chain = chain_objs[cid]
            for entry in by_chain.pop(cid):
                residue = self.modeller.topology.addResidue(entry["residue_name"], chain, id=entry["residue_id"])
                atom_objs = {}
                for name, elem in zip(entry["atom_names"], entry["atom_elements"]):
                    atom_objs[name] = self.modeller.topology.addAtom(name, elem, residue)
                for pos in entry["positions"]:
                    self.modeller.positions.append(pos)
                for a1, a2 in entry.get("bonds", []):
                    if a1 in atom_objs and a2 in atom_objs:
                        self.modeller.topology.addBond(atom_objs[a1], atom_objs[a2])

        # Then, create any missing chains and add their residues.
        for cid, entries in by_chain.items():
            chain = self.modeller.topology.addChain(cid)
            for entry in entries:
                residue = self.modeller.topology.addResidue(entry["residue_name"], chain, id=entry["residue_id"])
                atom_objs = {}
                for name, elem in zip(entry["atom_names"], entry["atom_elements"]):
                    atom_objs[name] = self.modeller.topology.addAtom(name, elem, residue)
                for pos in entry["positions"]:
                    self.modeller.positions.append(pos)
                for a1, a2 in entry.get("bonds", []):
                    if a1 in atom_objs and a2 in atom_objs:
                        self.modeller.topology.addBond(atom_objs[a1], atom_objs[a2])

    def addHydrogens(self, variants=None):
        # Create protein-only state and cache non-protein residues completely
        # (chain id, residue id, atom names/elements, positions) so we can
        # reinsert them verbatim after adding hydrogens to the protein.
        saved_residues, non_protein = self._extract_non_protein_residues()

        # Remove non-protein residues so hydrogens are added only to the protein
        if non_protein:
            self.modeller.delete(non_protein)

        # Add hydrogens to the remaining (protein) part
        self.modeller.addHydrogens(self.forcefield, variants=variants)

        # Reinsert the cached non-protein residues.
        self._reinsert_cached_residues(saved_residues)
        self._refresh_positions_cache()

    def addMembrane(self, membrane_system=None, platform=None):
        membrane_config = _normalize_membrane_system(
            membrane_system if membrane_system is not None else self.membrane_system
        )
        if membrane_config is None:
            raise ValueError("A membrane_system configuration is required to build a membrane.")

        if self._membrane_built:
            if membrane_config != self.membrane_system:
                raise ValueError(
                    "A membrane has already been added to this modeller with a different membrane_system."
                )
            return

        if not hasattr(self, "forcefield"):
            self.setUpFF("amber14", membrane_system=membrane_config)
        elif 'amber14/lipid17.xml' not in getattr(self, "ff_files", []):
            self.setUpFF(self.ff_name, membrane_system=membrane_config)
        else:
            self.membrane_system = membrane_config

        saved_chain_subsystem, excluded_chain_residues = self._extract_chain_subsystem(
            membrane_config.get("exclude_chain_ids", ())
        )
        if excluded_chain_residues:
            self.modeller.delete(excluded_chain_residues)

        saved_residues, non_protein = self._extract_non_protein_residues()
        if non_protein:
            self.modeller.delete(non_protein)

        pre_membrane_modeller = Modeller(self.modeller.topology, self.modeller.positions)
        last_exception = None
        for attempt in range(membrane_config["build_retries"]):
            if attempt > 0:
                self.modeller = Modeller(pre_membrane_modeller.topology, pre_membrane_modeller.positions)
            try:
                self.modeller.addMembrane(
                    self.forcefield,
                    lipidType=membrane_config["openmm_lipid_type"],
                    membraneCenterZ=membrane_config["membrane_center_z_nm"] * nanometer,
                    minimumPadding=membrane_config["minimum_padding_nm"] * nanometer,
                    positiveIon=membrane_config["positive_ion"],
                    negativeIon=membrane_config["negative_ion"],
                    ionicStrength=membrane_config["ionic_strength_molar"] * molar,
                    neutralize=membrane_config["neutralize"],
                    platform=platform,
                )
                last_exception = None
                break
            except Exception as exc:
                last_exception = exc
                if "nan" not in str(exc).lower() or attempt + 1 >= membrane_config["build_retries"]:
                    raise
        if last_exception is not None:
            raise last_exception
        if saved_chain_subsystem is not None:
            self.modeller.add(*saved_chain_subsystem)
        self._reinsert_cached_residues(saved_residues)
        self._membrane_built = True
        self._refresh_positions_cache()

    def getProtonationStates(self, keep_ligands=False):
        """
        Get residue names according to the protonation state.
        """

        residue_names = []
        for residue in self.modeller.topology.residues():

            if residue.name == 'HIS':
                atoms = []
                for atom in residue.atoms():
                    atoms.append(atom.name)
                if 'HD1' in atoms and 'HE2' in atoms:
                    his_name = 'HIP'
                elif 'HD1' in atoms:
                    his_name = 'HID'
                elif 'HE2' in atoms:
                    his_name = 'HIE'
                else:
                    his_name = None
                residue_names.append(his_name)

            elif residue.name == 'ASP':
                atoms = []
                for atom in residue.atoms():
                    atoms.append(atom.name)
                if 'HD2' in atoms:
                    residue_names.append('ASH')
                else:
                    residue_names.append('ASP')

            elif residue.name == 'GLU':
                atoms = []
                for atom in residue.atoms():
                    atoms.append(atom.name)
                if 'HE2' in atoms:
                    residue_names.append('GLH')
                else:
                    residue_names.append('GLU')

            elif residue.name == 'CYS':
                atoms = []
                for atom in residue.atoms():
                    atoms.append(atom.name)
                if 'HG' in atoms:
                    residue_names.append('CYS')
                else:
                    residue_names.append('CYX')
            else:
                if residue.name not in set(aa3)-set(ions) and not keep_ligands:
                    continue
                residue_names.append(None)

        return residue_names

    def prepareMCPBSite(self, output_folder, mcpb_site, overwrite=False, force_field="ff14SB"):
        """Prepare MCPB site artifacts without running the QM/MCPB pipeline."""
        normalized_site = _normalize_preparable_mcpb_site(mcpb_site)
        resolved_site_config = _resolve_mcpb_config(self.modeller.topology, self.modeller.positions, normalized_site)
        resolved_site = resolved_site_config["sites"][0]
        selection = _collect_mcpb_nonprotein_residue_selection(resolved_site_config)

        fragment_by_key = {}
        for fragment in normalized_site["fragments"]:
            key = (
                fragment["chain_id"],
                fragment["residue_id"],
                fragment["residue_name"],
            )
            if key in fragment_by_key:
                raise ValueError(
                    f"MCPB site '{normalized_site['site_id']}' defines the same fragment more than once: {key}."
                )
            fragment_by_key[key] = fragment

        missing_fragments = []
        for residue_key in selection["keys"]:
            if residue_key not in fragment_by_key:
                missing_fragments.append(f"{residue_key[2]} {residue_key[0]} {residue_key[1]}")
        if missing_fragments:
            raise ValueError(
                "prepareMCPBSite requires fragment definitions with net charges for every non-protein residue in the site. "
                "Missing fragments for: " + ", ".join(missing_fragments)
            )

        extra_fragments = []
        for residue_key in fragment_by_key:
            if residue_key not in selection["keys"]:
                extra_fragments.append(f"{residue_key[2]} {residue_key[0]} {residue_key[1]}")
        if extra_fragments:
            raise ValueError(
                "prepareMCPBSite received fragment definitions that are not part of the selected site: "
                + ", ".join(extra_fragments)
            )

        metal_residue_key = (
            resolved_site["metal"]["chain_id"],
            resolved_site["metal"]["residue_id"],
            resolved_site["metal"]["residue_name"],
        )
        metal_fragment = fragment_by_key.get(metal_residue_key)
        if resolved_site["metal"].get("is_embedded_metal", False):
            if metal_fragment is None:
                raise ValueError(
                    "Embedded-metal sites require a fragment definition for the metal-containing residue "
                    f"{metal_residue_key[2]} {metal_residue_key[0]} {metal_residue_key[1]}."
                )
            metal_atom_name = resolved_site["metal"]["atom_name"]
            if metal_atom_name not in metal_fragment["exclude_atoms"]:
                raise ValueError(
                    "Embedded-metal fragment definitions must exclude the metal atom so it can be written as a separate "
                    f"residue for MCPB. Add '{metal_atom_name}' to exclude_atoms for {metal_fragment['residue_name']}."
                )

        site_directory = os.path.join(output_folder, normalized_site["site_id"])
        fragments_directory = os.path.join(site_directory, "fragments")
        if os.path.exists(site_directory):
            if not overwrite:
                raise FileExistsError(
                    f"MCPB site directory {site_directory} already exists. Use overwrite=True to replace it."
                )
            shutil.rmtree(site_directory)
        os.makedirs(fragments_directory, exist_ok=True)

        fragment_manifest = []
        for residue_key in selection["keys"]:
            fragment = fragment_by_key[residue_key]
            residue = _find_residue_in_topology(
                self.modeller.topology,
                fragment["chain_id"],
                fragment["residue_id"],
                fragment["residue_name"],
                structure_residue_name=fragment.get("structure_residue_name"),
            )
            fragment_pdb_path = os.path.join(fragments_directory, f"{fragment['output_name']}.pdb")
            _write_filtered_residue_pdb(
                self.modeller,
                residue,
                fragment_pdb_path,
                exclude_atoms=fragment["exclude_atoms"],
                residue_name=fragment["output_name"],
            )
            fragment_manifest.append(
                {
                    "chain_id": fragment["chain_id"],
                    "residue_id": fragment["residue_id"],
                    "residue_name": fragment["residue_name"],
                    "structure_residue_name": fragment.get("structure_residue_name", fragment["residue_name"]),
                    "output_name": fragment["output_name"],
                    "net_charge": fragment["net_charge"],
                    "exclude_atoms": list(fragment["exclude_atoms"]),
                    "pdb_file": os.path.basename(fragment_pdb_path),
                }
            )

        metal_fragment_pdb = os.path.join(
            fragments_directory,
            f"{normalized_site['metal']['output_residue_name']}.pdb",
        )
        _write_single_atom_pdb(
            resolved_site["metal"],
            metal_fragment_pdb,
            residue_name=normalized_site["metal"]["output_residue_name"],
            atom_name=normalized_site["metal"]["output_atom_name"],
        )

        split_modeller, split_metal_spec = _build_split_mcpb_modeller(
            self.modeller,
            resolved_site,
            normalized_site["metal"]["output_residue_name"],
            normalized_site["metal"]["output_atom_name"],
        )

        split_site_core = {
            "site_id": normalized_site["site_id"],
            "group_name": normalized_site["group_name"],
            "metal": split_metal_spec,
            "coordinating_atoms": normalized_site["coordinating_atoms"],
            "cut_off": normalized_site["cut_off"],
        }
        split_resolved_site_config = _resolve_mcpb_config(
            split_modeller.topology,
            split_modeller.positions,
            split_site_core,
        )
        split_resolved_site = split_resolved_site_config["sites"][0]

        full_structure_pdb = os.path.join(site_directory, f"{normalized_site['site_id']}_mcpb_model.pdb")
        with open(full_structure_pdb, "w") as handle:
            PDBFile.writeFile(split_modeller.topology, split_modeller.positions, handle)
        _renumber_pdb_residues_globally(full_structure_pdb)

        topology_atom_count = 0
        for _ in split_modeller.topology.atoms():
            topology_atom_count += 1
        serial_by_atom_index = _pdb_atom_serials_by_topology_index(
            full_structure_pdb,
            expected_atom_count=topology_atom_count,
        )
        input_file = os.path.join(site_directory, f"{normalized_site['site_id']}.in")
        ion_ids = [serial_by_atom_index[split_resolved_site["metal"]["atom_index"]]]
        bonded_pairs = [
            (
                serial_by_atom_index[split_resolved_site["metal"]["atom_index"]],
                serial_by_atom_index[atom_record["atom_index"]],
            )
            for atom_record in split_resolved_site["coordinating_atoms"]
        ]
        naa_mol2files = [f"{fragment['output_name']}.mol2" for fragment in normalized_site["fragments"]]
        frcmod_files = [f"{fragment['output_name']}.frcmod" for fragment in normalized_site["fragments"]]
        ion_mol2files = [f"{normalized_site['metal']['output_residue_name']}.mol2"]

        with open(input_file, "w") as handle:
            handle.write(f"original_pdb {os.path.basename(full_structure_pdb)}\n")
            handle.write(f"group_name {normalized_site['group_name']}\n")
            handle.write(f"software_version {normalized_site['qm']['software_version']}\n")
            handle.write(f"force_field {force_field}\n")
            handle.write(f"cut_off {normalized_site['cut_off']}\n")
            handle.write("ion_ids " + " ".join(str(atom_id) for atom_id in ion_ids) + "\n")
            handle.write("ion_mol2files " + " ".join(ion_mol2files) + "\n")
            if naa_mol2files:
                handle.write("naa_mol2files " + " ".join(naa_mol2files) + "\n")
            if frcmod_files:
                handle.write("frcmod_files " + " ".join(frcmod_files) + "\n")
            if bonded_pairs:
                handle.write(
                    "add_bonded_pairs "
                    + " ".join(f"{atom_1}-{atom_2}" for atom_1, atom_2 in bonded_pairs)
                    + "\n"
                )
            handle.write(f"smmodel_chg {normalized_site['qm']['small_model_charge']}\n")
            handle.write(f"smmodel_spin {normalized_site['qm']['small_model_spin']}\n")
            handle.write(f"lgmodel_chg {normalized_site['qm']['large_model_charge']}\n")
            handle.write(f"lgmodel_spin {normalized_site['qm']['large_model_spin']}\n")

        commands_file = os.path.join(site_directory, "prepare_fragments.sh")
        command_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "cd \"$(dirname \"$0\")\"",
            f"metalpdb2mol2.py -i fragments/{normalized_site['metal']['output_residue_name']}.pdb "
            f"-o {normalized_site['metal']['output_residue_name']}.mol2 -c {normalized_site['metal']['formal_charge']}",
        ]
        for fragment in normalized_site["fragments"]:
            command_lines.append(
                f"antechamber -i fragments/{fragment['output_name']}.pdb -fi pdb "
                f"-o {fragment['output_name']}.mol2 -fo mol2 -c bcc -nc {fragment['net_charge']} "
                f"-rn {fragment['output_name']} -pf y"
            )
            command_lines.append(
                f"parmchk2 -i {fragment['output_name']}.mol2 -o {fragment['output_name']}.frcmod -f mol2"
            )
        command_lines.extend(
            [
                f"MCPB.py -i {os.path.basename(input_file)} -s 1",
                "# Run the generated QM jobs, then continue with:",
                f"# MCPB.py -i {os.path.basename(input_file)} -s 2",
                f"# MCPB.py -i {os.path.basename(input_file)} -s 3",
                f"# MCPB.py -i {os.path.basename(input_file)} -s 4",
            ]
        )
        with open(commands_file, "w") as handle:
            handle.write("\n".join(command_lines) + "\n")
        try:
            os.chmod(commands_file, 0o755)
        except OSError:
            pass

        manifest = {
            "site_id": normalized_site["site_id"],
            "group_name": normalized_site["group_name"],
            "site_directory": site_directory,
            "full_structure_pdb": os.path.basename(full_structure_pdb),
            "input_file": os.path.basename(input_file),
            "commands_file": os.path.basename(commands_file),
            "metal": {
                "original_residue_name": resolved_site["metal"]["residue_name"],
                "original_chain_id": resolved_site["metal"]["chain_id"],
                "original_residue_id": resolved_site["metal"]["residue_id"],
                "atom_name": resolved_site["metal"]["atom_name"],
                "formal_charge": normalized_site["metal"]["formal_charge"],
                "output_residue_name": normalized_site["metal"]["output_residue_name"],
                "output_atom_name": normalized_site["metal"]["output_atom_name"],
                "split_chain_id": split_metal_spec["chain_id"],
                "split_residue_id": split_metal_spec["residue_id"],
            },
            "qm": dict(normalized_site["qm"]),
            "fragments": fragment_manifest,
            "ion_ids": ion_ids,
            "add_bonded_pairs": bonded_pairs,
            "naa_mol2files": naa_mol2files,
            "frcmod_files": frcmod_files,
        }
        manifest_path = os.path.join(site_directory, "site_manifest.json")
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        manifest["manifest_file"] = os.path.basename(manifest_path)
        return manifest

    def addSolvent(self):
        self.modeller.addSolvent(self.forcefield)
        self._refresh_positions_cache()

    def parameterizePDBLigands(self, parameters_folder, charges=None, skip_ligands=None, overwrite=False,
                               metal_ligand=None, add_bonds=None, cpus=None, return_qm_jobs=False,
                               extra_force_field=None,charge_model='bcc',
                               force_field='ff14SB', residue_names=None, metal_parameters=None, extra_frcmod=None,
                               extra_mol2=None, add_counterions=True, add_counterionsRand=False, save_amber_pdb=False, solvate=True,
                               regenerate_amber_files=False, non_standard_residues=None, strict_atom_name_check=True,
                               only_residues=None, build_full_system=True, skip_ligand_charge_computation=False,
                               ligand_sdf_files=None, export_per_residue_ffxml=False,
                               run_acdoctor=True,
                               ligand_xml_files=None, solvatebox_buffer=12.0, solvatebox_iso=False,
                               mcpb_config=None, mcpb_site=None, membrane_system=None):

        def _changeGaussianCPUS(gaussian_com_file, cpus):
            tmp = open('file.tmp', 'w')
            with open(gaussian_com_file) as gf:
                for l in gf:
                    if l.startswith('%NProcShared='):
                         l = l.strip()[:-1]+str(cpus)+'\n'
                    tmp.write(l)
            tmp.close()
            shutil.move('file.tmp', gaussian_com_file)

        def _checkIfGaussianFinished(gaussian_log_file):
            finished = False
            if os.path.exists(gaussian_log_file):
                with open(gaussian_log_file) as lf:
                    for l in lf:
                        if 'Normal termination of Gaussian 09' in l:
                            finished = True
            return finished

        def getNonProteinResidues(topology, skip_residues=None):
            residues = []
            for chain in topology.chains():
                for r in chain.residues():
                    residue = r.name.upper()
                    if residue not in aa3:
                        #Skip water molecules
                        if residue == 'HOH':
                            continue
                        # Skip given ligands
                        if residue in skip_residues:
                            continue
                        residues.append(r)
            return residues

        def getResiduesByName(topology, residue_name):
            residues = []
            for chain in topology.chains():
                for r in chain.residues():
                    if r.name.upper() == residue_name:
                        residues.append(r)
            return residues

        def getMissingAtomTypes(mol2):
            cond = False
            missing = []
            with open(mol2) as mf:
                for l in mf:
                    if l.startswith('@<TRIPOS>ATOM'):
                        cond = True
                        continue
                    elif l.startswith('@<TRIPOS>BOND'):
                        cond = False
                        continue
                    if cond:
                        if l.split()[5].isupper():
                            missing.append(l.split()[5])
            return missing

        def getAtomTypes(frcmod, atoms):
            cond = False
            atom_types = {}
            with open(frcmod) as f:
                for l in f:
                    if l.startswith('MASS'):
                        cond = True
                        continue
                    elif l.startswith('BOND'):
                        cond = False
                        continue
                    if cond:
                        if l.strip() == '':
                            continue
                        if l.split()[3] == 'ion':
                            atom_types[l.split()[0]] = l.split()[2]
                        elif 'disulfide' in l:
                            atom_types[l.split()[0]] = l.split()[3]
                        else:
                            atom_types[l.split()[0]] = l.split()[4]
            return atom_types

        def formatSolvateboxBuffer(buffer):
            """
            Return a tleap-compatible solvatebox buffer expression (Angstrom).
            Accepts a scalar or a 3-element iterable (x/y/z buffers).
            """
            if isinstance(buffer, (int, float)):
                value = float(buffer)
                if value <= 0.0:
                    raise ValueError("solvatebox_buffer scalar must be > 0.")
                return f"{value:.6f}"

            if isinstance(buffer, (list, tuple, np.ndarray)):
                if len(buffer) != 3:
                    raise ValueError("solvatebox_buffer iterable must have exactly 3 elements.")
                values = [float(v) for v in buffer]
                if any(v <= 0.0 for v in values):
                    raise ValueError("All solvatebox_buffer elements must be > 0.")
                return "{" + " ".join(f"{v:.6f}" for v in values) + "}"

            raise TypeError(
                "solvatebox_buffer must be a positive float or a 3-element iterable of positive floats."
            )

        def _extract_pdb_atom_names(pdb_path):
            atom_names = set()
            if not os.path.exists(pdb_path):
                return atom_names
            with open(pdb_path) as handle:
                for line in handle:
                    if line.startswith(('ATOM', 'HETATM')):
                        name = line[12:16].strip()
                        if name:
                            atom_names.add(name.upper())
            return atom_names

        def _validate_ligand_pdb_atoms(generated_path, reference_path, residue_name, source_label):
            if not os.path.exists(generated_path) or not os.path.exists(reference_path):
                return
            generated_atoms = _extract_pdb_atom_names(generated_path)
            reference_atoms = _extract_pdb_atom_names(reference_path)
            if not generated_atoms and not reference_atoms:
                return
            missing = reference_atoms - generated_atoms
            extra = generated_atoms - reference_atoms
            if missing or extra:
                mismatch = []
                if missing:
                    mismatch.append(f"missing atoms {sorted(missing)}")
                if extra:
                    mismatch.append(f"extra atoms {sorted(extra)}")
                message = (
                    f"Atom name mismatch for ligand {residue_name}: generated PDB "
                    f"({generated_path}) differs from {source_label} ({reference_path}); "
                    + ", ".join(mismatch)
                )
                if strict_atom_name_check:
                    raise ValueError(message)
                warnings.warn(message, UserWarning)

        if only_residues:
            if isinstance(only_residues, str):
                only_residue_set = {only_residues.strip().upper()} if only_residues.strip() else set()
            else:
                only_residue_set = {str(res).strip().upper() for res in only_residues if str(res).strip()}
            if not only_residue_set:
                only_residue_set = None
        else:
            only_residue_set = None

        skip_ligand_charge_computation = bool(skip_ligand_charge_computation)

        if regenerate_amber_files:
            if os.path.exists(parameters_folder+'/'+self.pdb_name+'.prmtop'):
                os.remove(parameters_folder+'/'+self.pdb_name+'.prmtop')
            if os.path.exists(parameters_folder+'/'+self.pdb_name+'.inpcrd'):
                os.remove(parameters_folder+'/'+self.pdb_name+'.inpcrd')

        if os.path.exists(parameters_folder+'/'+self.pdb_name+'.prmtop') and os.path.exists(parameters_folder+'/'+self.pdb_name+'.inpcrd'):
            if not overwrite:

                # Set topology and positions to amber's
                self.prmtop_file = parameters_folder+'/'+self.pdb_name+'.prmtop'
                self.inpcrd_file = parameters_folder+'/'+self.pdb_name+'.inpcrd'

                prmtop = AmberPrmtopFile(self.prmtop_file)
                inpcrd = AmberInpcrdFile(self.inpcrd_file)

                self.modeller.topology = prmtop.topology
                self.modeller.positions = inpcrd.positions

                print('Parameters were already created. Give overwrite=True to recompute them.')
                return

        # Create working folder
        if not os.path.exists(parameters_folder):
            os.mkdir(parameters_folder)

        extra_ffs = ['parmBSC1', 'parmBSC2']
        if extra_force_field:
            if extra_force_field not in extra_ffs:
                raise ValueError(f'The only implemented extra ff is: {extra_ffs[0]}')

            # Copy ff files
            extra_ff_folder = parameters_folder+'/'+extra_force_field
            if not os.path.exists(extra_ff_folder):
                os.mkdir(extra_ff_folder)
            _copyFFFiles(extra_ff_folder, extra_force_field)

            extra_force_field_source = extra_ff_folder+'/leaprc.bsc'+extra_force_field[-1]

            # Define ff residues
            ff_residues = [
                            'A', 'A3', 'A5', 'AN',
                            'C', 'C3', 'C5', 'CN',
                            'DA', 'DA3', 'DA5', 'DAN',
                            'DC', 'DC3', 'DC5', 'DCN',
                            'DG', 'DG3', 'DG5', 'DGN',
                            'DT', 'DT3', 'DT5', 'DTN',
                            'G', 'G3', 'G5', 'GN',
                            'OHE',
                            'U', 'U3', 'U5', 'UN'
                        ]

            if skip_ligands:
                skip_ligands += ff_residues
            else:
                skip_ligands = ff_residues

        if not skip_ligands:
            skip_ligands = []
        else:
            skip_ligands = [str(residue).strip().upper() for residue in skip_ligands if str(residue).strip()]

        membrane_system = _normalize_membrane_system(
            membrane_system if membrane_system is not None else self.membrane_system
        )
        if membrane_system is not None:
            if not self._membrane_built:
                self.addMembrane(membrane_system)
            for residue_name in membrane_system["skip_residue_names"]:
                skip_ligands.append(residue_name)
            skip_ligands = _ordered_unique(skip_ligands)
            solvate = False
            add_counterions = False
            add_counterionsRand = False

        if not metal_ligand:
            metal_ligand = {}

        if mcpb_site is not None and mcpb_config is not None:
            raise ValueError("Use only one of mcpb_site or mcpb_config. mcpb_config is kept only as a compatibility alias.")
        if mcpb_config is not None:
            warnings.warn(
                "mcpb_config is deprecated; use mcpb_site instead. extra_mol2/extra_frcmod remain the path for "
                "reusing generated MCPB parameter files.",
                UserWarning,
            )
        normalized_mcpb_site = mcpb_site if mcpb_site is not None else mcpb_config
        if metal_ligand and normalized_mcpb_site is None:
            raise ValueError(
                "MCPB parameterization now requires an explicit mcpb_site definition when metal_ligand is used. "
                "Keep extra_mol2/extra_frcmod for reusing pre-generated mol2/frcmod files."
            )

        resolved_mcpb_config = _resolve_mcpb_config(self.modeller.topology, self.modeller.positions, normalized_mcpb_site)
        self.mcpb_site = resolved_mcpb_config
        self.mcpb_config = resolved_mcpb_config
        self.mcpb_site_definitions = resolved_mcpb_config["sites"] if resolved_mcpb_config else []
        mcpb_site_residue_selection = _collect_mcpb_nonprotein_residue_selection(resolved_mcpb_config)
        self.mcpb_nonprotein_residue_selection = mcpb_site_residue_selection
        if mcpb_site_residue_selection["duplicate_names"]:
            duplicate_summary = []
            for residue_name, entries in sorted(mcpb_site_residue_selection["duplicate_names"].items()):
                locations = ", ".join(
                    f"{entry['residue_name']} {entry['chain_id']} {entry['residue_id']}"
                    for entry in entries
                )
                duplicate_summary.append(f"{residue_name}: {locations}")
            raise ValueError(
                "mcpb_config currently selects multiple instances of the same non-protein residue name. "
                "This workflow still stages non-protein parameter files by residue name, so ambiguous multi-instance "
                "sites must be split into separate runs. Conflicts: "
                + "; ".join(duplicate_summary)
            )
        use_site_driven_mcpb = bool(resolved_mcpb_config)
        if resolved_mcpb_config and not metal_ligand:
            legacy_metal_ligand = _legacy_metal_ligand_from_mcpb_config(resolved_mcpb_config)
            if legacy_metal_ligand:
                metal_ligand = legacy_metal_ligand
        use_mcpb = bool(metal_ligand) or use_site_driven_mcpb

        # Generate set of metal ligand values
        metal_ligand_values = []
        for r in metal_ligand:
            if isinstance(metal_ligand[r], str): # Convert into a list
                metal_ligand[r] = [metal_ligand[r]]
            for m in metal_ligand[r]:
                metal_ligand_values.append(m)

        # Get parameter folders from the given metal_parameters folder
        parameters_folders = {}
        parameters_mol2 = {}
        parameters_frcmod = {}
        if metal_parameters:
            parameters_folders = {}
            for d in os.listdir(metal_parameters):
                if d.endswith('_parameters'):
                    parameters_folders[d.split('_')[0]] = metal_parameters+'/'+d
                elif d.endswith('_mcpbpy.frcmod'):
                    parameters_folders['mcpbpy.frcmod'] = metal_parameters+'/'+d
                elif d.endswith('.mol2'):
                    parameters_mol2[d.split('.')[0]] = metal_parameters+'/'+d
                elif d.endswith('.frcmod'):
                    parameters_frcmod[d.split('.')[0]] = metal_parameters+'/'+d

        if not cpus:
            cpus = cpu_count()

        # Normalize ligand_xml_files mapping (RESNAME -> ffxml path)
        normalized_ligand_xml = {}
        if ligand_xml_files:
            if not isinstance(ligand_xml_files, Mapping):
                raise TypeError("ligand_xml_files must be a mapping of residue name -> FFXML path.")
            for key, value in ligand_xml_files.items():
                if not isinstance(key, str):
                    raise TypeError("ligand_xml_files keys must be residue name strings.")
                resname = key.strip().upper()
                if not resname:
                    continue
                normalized_ligand_xml[resname] = os.fspath(value)

        if residue_names is not None and isinstance(residue_names, Mapping) and len(residue_names) == 0:
            warnings.warn(
                "residue_names was provided as an empty mapping; no explicit protonation-state residue names "
                "will be applied.",
                UserWarning,
            )

        # Modify protonation state names and save state
        for chain in self.modeller.topology.chains():
            for i,residue in enumerate(chain.residues()):

                res_tuple = (chain.id, int(residue.id))
                if residue_names and res_tuple in residue_names:
                    residue.name = residue_names[res_tuple]

                if residue.name not in aa3:
                    continue

                if i == 0:
                    for atom in residue.atoms():
                        if atom.name == 'H':
                            atom.name = 'H1'

                if residue.name == 'HIS':
                    atoms = []
                    for atom in residue.atoms():
                        atoms.append(atom.name)
                    if 'HD1' in atoms and 'HE2' in atoms:
                        his_name = 'HIP'
                    elif 'HD1' in atoms:
                        his_name = 'HID'
                    elif 'HE2' in atoms:
                        his_name = 'HIE'
                    else:
                        his_name = 'HIS'
                    residue.name = his_name

                elif residue.name == 'ASP':
                    atoms = []
                    for atom in residue.atoms():
                        atoms.append(atom.name)
                    if 'HD2' in atoms:
                        residue.name = 'ASH'
                    else:
                        residue.name = 'ASP'

                elif residue.name == 'GLU':
                    atoms = []
                    for atom in residue.atoms():
                        atoms.append(atom.name)
                    if 'HE2' in atoms:
                        residue.name = 'GLH'
                    else:
                        residue.name = 'GLU'

                elif residue.name == 'CYS':
                    atoms = []
                    for atom in residue.atoms():
                        atoms.append(atom.name)
                    if 'HG' in atoms:
                        residue.name = 'CYS'
                    else:
                        residue.name = 'CYX'

        pdb_file = parameters_folder+'/'+self.pdb_name+'.pdb'
        self.savePDB(pdb_file)

        # Get molecules that need parameterization
        par_folder = {}
        generated_pdb_paths = {}
        ffxml_converted = set()

        for r in getNonProteinResidues(self.modeller.topology, skip_residues=skip_ligands):

            residue = r.name.upper()
            if only_residue_set and residue not in only_residue_set:
                continue
            if mcpb_site_residue_selection["keys"]:
                residue_key = (r.chain.id, _normalize_residue_id(r.id), residue)
                if (
                    residue in mcpb_site_residue_selection["names"]
                    and residue_key not in mcpb_site_residue_selection["keys"]
                ):
                    continue

            # Create PDB for each ligand molecule
            lig_top, lig_pos = extract_residue_subsystem(self.modeller, r)

            par_folder[residue] = parameters_folder+'/'+residue+'_parameters'

            if not os.path.exists(par_folder[residue]):
                os.mkdir(par_folder[residue])

            generated_ligand_pdb = par_folder[residue]+'/'+residue+'.pdb'
            with open(generated_ligand_pdb, 'w') as rf:
                PDBFile.writeFile(lig_top, lig_pos, rf)
            generated_pdb_paths[residue] = generated_ligand_pdb
            model_specific_pdb = os.path.join(parameters_folder, f"{self.pdb_name}_{residue}.pdb")
            shutil.copyfile(generated_ligand_pdb, model_specific_pdb)

            if residue in parameters_folders:
                provided_pack = parameters_folders[residue]
                pack_pdb = os.path.join(provided_pack, residue+'.pdb')
                _validate_ligand_pdb_atoms(
                    generated_ligand_pdb,
                    pack_pdb,
                    residue,
                    f"parameter pack in {provided_pack}"
                )
                for d in os.listdir(parameters_folders[residue]):
                    _copyfile_if_needed(parameters_folders[residue]+'/'+d,
                                        par_folder[residue]+'/'+d)

            if residue in parameters_mol2:
                canonical_mol2 = os.path.join(parameters_folder, f'{residue}.mol2')
                _copyfile_if_needed(parameters_mol2[residue], canonical_mol2)

        # If FFXMLs are provided, generate mol2/frcmod for AmberTools (one per residue)
        if normalized_ligand_xml:
            try:
                import parmed as pmd
                from openmm.app import ForceField
            except Exception as exc:
                warnings.warn(f"[prepare_proteins] parmed/openmm unavailable for FFXML conversion: {exc}", RuntimeWarning)
            else:
                for resname, pdb_path in generated_pdb_paths.items():
                    if resname not in normalized_ligand_xml:
                        continue
                    ffxml_path = normalized_ligand_xml[resname]
                    mol2_path = os.path.join(parameters_folder, f"{resname}.mol2")
                    frcmod_path = os.path.join(parameters_folder, f"{resname}.frcmod")
                    if os.path.exists(mol2_path):
                        parameters_mol2[resname] = mol2_path
                        continue
                    try:
                        lig_pdb = PDBFile(pdb_path)
                        ff_local = ForceField(ffxml_path)
                        lig_sys = ff_local.createSystem(lig_pdb.topology)
                        lig_struct = pmd.openmm.load_topology(lig_pdb.topology, lig_sys, xyz=lig_pdb.positions)
                        lig_struct.save(mol2_path, overwrite=True)
                        parameters_mol2[resname] = mol2_path
                        # Derive a frcmod directly from the OpenMM system parameters
                        try:
                            param_set = pmd.amber.AmberParameterSet.from_structure(lig_struct)
                            param_set.write(frcmod_path, style="frcmod")
                            parameters_frcmod[resname] = frcmod_path
                            workdir = par_folder.get(resname)
                            if workdir and os.path.isdir(workdir):
                                dest = os.path.join(workdir, f"{resname}.frcmod")
                                param_set.write(dest, style="frcmod")
                        except Exception as exc:
                            warnings.warn(
                                f"[prepare_proteins] Failed to write frcmod from FFXML for {resname}: {exc}",
                                RuntimeWarning,
                            )
                        # marker to skip acdoctor for ffxml-generated MOL2
                        marker_path = os.path.join(parameters_folder, f".ffxml_generated_{resname.lower()}")
                        try:
                            with open(marker_path, "w") as mh:
                                mh.write("ffxml")
                        except OSError:
                            pass
                        # Also drop a marker inside the residue work folder if it exists
                        work_marker = os.path.join(par_folder.get(resname, parameters_folder), f".ffxml_generated_{resname.lower()}")
                        try:
                            with open(work_marker, "w") as mh:
                                mh.write("ffxml")
                        except Exception:
                            pass
                        ffxml_converted.add(resname.upper())
                    except Exception as exc:
                        warnings.warn(
                            f"[prepare_proteins] Failed to derive mol2/frcmod from {ffxml_path} for {resname}: {exc}",
                            RuntimeWarning,
                        )

        mol2_conversion_targets = set()
        provided_frcmod_residues = set()
        provided_mol2_residues = set()
        residue_frcmod_files = defaultdict(list)
        residue_mol2_files = defaultdict(list)

        def _register_frcmod_file(residue_name, path):
            if not path:
                return
            res_upper = residue_name.upper()
            files = residue_frcmod_files[res_upper]
            if path not in files:
                files.append(path)
            return files

        def _register_mol2_file(residue_name, path):
            if not path:
                return None
            res_upper = residue_name.upper()
            files = residue_mol2_files[res_upper]
            if path not in files:
                files.append(path)
            return files

        def _copy_into_par_folder(residue_name, source_path):
            """Duplicate a provided parameter file into the residue-specific workspace."""
            if not source_path or not os.path.exists(source_path):
                return None
            res_upper = residue_name.upper()
            if res_upper not in par_folder:
                return None
            basename = os.path.basename(os.fspath(source_path))
            destination = os.path.join(par_folder[res_upper], basename)
            _copyfile_if_needed(source_path, destination)
            return destination

        def _get_case_insensitive(mapping, key):
            """Return a value from `mapping` regardless of key casing."""
            key_str = str(key)
            if key_str in mapping:
                return mapping[key_str]
            key_upper = key_str.upper()
            if key_upper in mapping:
                return mapping[key_upper]
            key_lower = key_str.lower()
            if key_lower in mapping:
                return mapping[key_lower]
            return None

        def _stage_external_mol2(residue_name, source_path, mark_for_conversion=False):
            """Place a provided MOL2 inside the working folder so antechamber can reuse it."""
            res_upper = residue_name.upper()
            if res_upper not in par_folder:
                return None
            if not os.path.exists(source_path):
                raise ValueError(f'Mol2 file for {residue_name} at {source_path} was not found.')
            workdir = par_folder[res_upper]
            basename = os.path.basename(os.fspath(source_path))
            dest_path = os.path.join(workdir, basename)
            try:
                if os.path.exists(dest_path) and os.path.samefile(source_path, dest_path):
                    return dest_path
            except FileNotFoundError:
                pass
            shutil.copyfile(source_path, dest_path)
            canonical = os.path.join(workdir, f'{res_upper}.mol2')
            if os.path.normpath(dest_path) != os.path.normpath(canonical) and not os.path.exists(canonical):
                try:
                    shutil.copyfile(dest_path, canonical)
                except OSError:
                    pass
            if mark_for_conversion:
                mol2_conversion_targets.add(res_upper)
            return dest_path

        def _extract_sdf_block(source_path, dest_path, index=0):
            """Write the selected molecule block from an SDF into dest_path."""
            index = int(index) if index else 0
            current = []
            blocks = []
            with open(source_path) as sf:
                for line in sf:
                    if line.strip() == "$$$$":
                        blocks.append("".join(current))
                        current = []
                    else:
                        current.append(line)
                if current:
                    blocks.append("".join(current))
            if not blocks:
                raise ValueError(f"No molecules found in SDF {source_path}")
            if index < 0 or index >= len(blocks):
                raise IndexError(
                    f"SDF index {index} out of range for {source_path}; found {len(blocks)} entries."
                )
            content = blocks[index]
            if not content.endswith("\n"):
                content += "\n"
            with open(dest_path, "w") as df:
                df.write(content)
                df.write("$$$$\n")

        for residue_name in par_folder:
            canonical_mol2 = os.path.join(parameters_folder, f"{residue_name}.mol2")
            if os.path.exists(canonical_mol2):
                _register_mol2_file(residue_name, canonical_mol2)

        for residue_name in par_folder:
            if residue_name in parameters_mol2:
                _stage_external_mol2(residue_name, parameters_mol2[residue_name])

        if extra_frcmod:
            def _iter_extra_frcmod_sources(values):
                """Yield (residue, source_entry) pairs preserving the user-provided order."""
                if isinstance(values, Mapping):
                    for res_name, src_values in values.items():
                        if isinstance(src_values, (list, tuple)):
                            for item in src_values:
                                yield res_name, item
                        else:
                            yield res_name, src_values
                else:
                    if isinstance(values, (list, tuple)):
                        iterable = values
                    elif isinstance(values, (str, os.PathLike)):
                        iterable = [values]
                    else:
                        iterable = [values]
                    for item in iterable:
                        yield None, item

            def _resolve_frcmod_source(residue_hint, entry):
                """Return (res_upper, source_path) for a user entry."""
                if entry is None:
                    raise ValueError('Frcmod file None was not found.')
                entry_path = os.fspath(entry) if isinstance(entry, os.PathLike) else entry
                source_path = None
                res_upper = None

                if residue_hint is not None:
                    res_upper = str(residue_hint).upper()

                if isinstance(entry, (str, os.PathLike)) and os.path.exists(entry_path):
                    source_path = entry_path
                    if res_upper is None:
                        res_upper = os.path.splitext(os.path.basename(entry_path))[0].upper()
                else:
                    lookup_key = entry_path
                    if residue_hint is not None:
                        source_path = _get_case_insensitive(parameters_frcmod, residue_hint)
                        if not source_path:
                            fallback = os.path.splitext(str(residue_hint))[0]
                            source_path = _get_case_insensitive(parameters_frcmod, fallback)
                    if not source_path:
                        source_path = _get_case_insensitive(parameters_frcmod, lookup_key)
                    if not source_path:
                        fallback = os.path.splitext(os.path.basename(lookup_key))[0]
                        source_path = _get_case_insensitive(parameters_frcmod, fallback)
                    if not source_path or not os.path.exists(source_path):
                        location_msg = f' at {entry_path}' if isinstance(entry, (str, os.PathLike)) and not os.path.exists(entry_path) else ''
                        if metal_parameters:
                            raise ValueError(f'Frcmod file for residue {residue_hint or entry_path} was not found in {metal_parameters}{location_msg}.')
                        raise ValueError(f'Frcmod file for residue {residue_hint or entry_path}{location_msg} was not found.')
                    if res_upper is None:
                        res_upper = os.path.splitext(os.path.basename(source_path))[0].upper()

                return res_upper, source_path

            for residue_key, source in _iter_extra_frcmod_sources(extra_frcmod):
                res_upper, source_path = _resolve_frcmod_source(residue_key, source)
                basename = os.path.basename(source_path)
                dest_root = os.path.join(parameters_folder, basename)
                copy_required = True
                if os.path.exists(dest_root):
                    same_file = False
                    try:
                        same_file = os.path.samefile(source_path, dest_root)
                    except FileNotFoundError:
                        same_file = False
                    if same_file:
                        copy_required = False
                    else:
                        try:
                            if cmp(source_path, dest_root, shallow=False):
                                copy_required = False
                            else:
                                raise ValueError(
                                    f'Conflicting frcmod destination {dest_root} already exists with different contents.'
                                )
                        except OSError:
                            raise ValueError(
                                f'Unable to compare frcmod destination {dest_root} with source {source_path}.'
                            )
                if copy_required:
                    _copyfile_if_needed(source_path, dest_root)
                canonical_root = os.path.join(parameters_folder, f"{res_upper}.frcmod")
                if os.path.normpath(dest_root) != os.path.normpath(canonical_root) and not os.path.exists(canonical_root):
                    _copyfile_if_needed(dest_root, canonical_root)
                _copy_into_par_folder(res_upper, dest_root)
                _register_frcmod_file(res_upper, dest_root)
                provided_frcmod_residues.add(res_upper)

        if extra_mol2:
            def _iter_extra_mol2_sources(values):
                if isinstance(values, Mapping):
                    for res_name, src_values in values.items():
                        if isinstance(src_values, (list, tuple)):
                            for item in src_values:
                                yield res_name, item
                        else:
                            yield res_name, src_values
                else:
                    if isinstance(values, (list, tuple)):
                        iterable = values
                    elif isinstance(values, (str, os.PathLike)):
                        iterable = [values]
                    else:
                        iterable = [values]
                    for item in iterable:
                        yield None, item

            def _resolve_mol2_source(residue_hint, entry):
                if entry is None:
                    raise ValueError('Mol2 file None was not found.')
                entry_path = os.fspath(entry) if isinstance(entry, os.PathLike) else entry
                source_path = None
                res_upper = None
                from_user = False

                if residue_hint is not None:
                    res_upper = str(residue_hint).upper()

                if isinstance(entry, (str, os.PathLike)) and os.path.exists(entry_path):
                    source_path = entry_path
                    from_user = True
                    if res_upper is None:
                        res_upper = os.path.splitext(os.path.basename(entry_path))[0].upper()
                else:
                    lookup_key = entry_path
                    if residue_hint is not None:
                        source_path = _get_case_insensitive(parameters_mol2, residue_hint)
                        if not source_path:
                            fallback = os.path.splitext(str(residue_hint))[0]
                            source_path = _get_case_insensitive(parameters_mol2, fallback)
                    if not source_path:
                        source_path = _get_case_insensitive(parameters_mol2, lookup_key)
                    if not source_path:
                        fallback = os.path.splitext(os.path.basename(lookup_key))[0]
                        source_path = _get_case_insensitive(parameters_mol2, fallback)
                    if not source_path or not os.path.exists(source_path):
                        location_msg = f' at {entry_path}' if isinstance(entry, (str, os.PathLike)) and not os.path.exists(entry_path) else ''
                        if metal_parameters:
                            raise ValueError(f'Mol2 file for residue {residue_hint or entry_path} was not found in {metal_parameters}{location_msg}.')
                        raise ValueError(f'Mol2 file for residue {residue_hint or entry_path}{location_msg} was not found.')
                    if res_upper is None:
                        res_upper = os.path.splitext(os.path.basename(source_path))[0].upper()

                return res_upper, source_path, from_user

            for residue_key, source in _iter_extra_mol2_sources(extra_mol2):
                res_upper, source_path, source_is_user = _resolve_mol2_source(residue_key, source)
                basename = os.path.basename(source_path)
                dest_root_mol2 = os.path.join(parameters_folder, basename)
                copy_required = True
                if os.path.exists(dest_root_mol2):
                    same_file = False
                    try:
                        same_file = os.path.samefile(source_path, dest_root_mol2)
                    except FileNotFoundError:
                        same_file = False
                    if same_file:
                        copy_required = False
                    else:
                        try:
                            if cmp(source_path, dest_root_mol2, shallow=False):
                                copy_required = False
                            else:
                                raise ValueError(
                                    f'Conflicting mol2 destination {dest_root_mol2} already exists with different contents.'
                                )
                        except OSError:
                            raise ValueError(
                                f'Unable to compare mol2 destination {dest_root_mol2} with source {source_path}.'
                            )
                if copy_required:
                    _copyfile_if_needed(source_path, dest_root_mol2)
                canonical_root = os.path.join(parameters_folder, f"{res_upper}.mol2")
                if os.path.normpath(dest_root_mol2) != os.path.normpath(canonical_root) and not os.path.exists(canonical_root):
                    _copyfile_if_needed(dest_root_mol2, canonical_root)
                _stage_external_mol2(res_upper, dest_root_mol2, mark_for_conversion=res_upper not in provided_frcmod_residues)
                provided_mol2_residues.add(res_upper)
                _register_mol2_file(res_upper, dest_root_mol2)
                generated_path = generated_pdb_paths.get(res_upper)
                if generated_path:
                    if source_is_user:
                        candidate_pdb = os.path.splitext(source_path)[0] + '.pdb'
                        if os.path.exists(candidate_pdb):
                            _validate_ligand_pdb_atoms(
                                generated_path,
                                candidate_pdb,
                                res_upper,
                                f"extra_mol2 source {source_path}"
                            )
                    else:
                        pack_pdb = os.path.join(os.path.dirname(source_path), f'{res_upper}.pdb')
                        if os.path.exists(pack_pdb):
                            _validate_ligand_pdb_atoms(
                                generated_path,
                                pack_pdb,
                                res_upper,
                                f"extra_mol2 metal_parameters source {source_path}"
                            )

        skip_parameterization_residues = provided_mol2_residues & provided_frcmod_residues
        if ffxml_converted:
            skip_parameterization_residues |= ffxml_converted

        if mcpb_site_residue_selection["embedded_residue_names"]:
            missing_embedded_assets = []
            for residue_name in mcpb_site_residue_selection["embedded_residue_names"]:
                mol2_path = _resolve_residue_parameter_file(
                    residue_name,
                    residue_mol2_files,
                    par_folder,
                    parameters_folder,
                    ".mol2",
                )
                frcmod_path = _resolve_residue_parameter_file(
                    residue_name,
                    residue_frcmod_files,
                    par_folder,
                    parameters_folder,
                    ".frcmod",
                )
                if mol2_path is None or frcmod_path is None:
                    missing_bits = []
                    if mol2_path is None:
                        missing_bits.append("mol2")
                    if frcmod_path is None:
                        missing_bits.append("frcmod")
                    missing_embedded_assets.append(f"{residue_name} ({'/'.join(missing_bits)})")
            if missing_embedded_assets:
                raise ValueError(
                    "Embedded-metal residues selected by mcpb_config cannot be parameterized with the generic ligand "
                    "workflow. Provide base parameter files via extra_mol2/extra_frcmod or metal_parameters for: "
                    + ", ".join(missing_embedded_assets)
                )

        # Normalize export_per_residue_ffxml target set
        export_ffxml_all = False
        export_ffxml_targets = set()
        if export_per_residue_ffxml:
            if export_per_residue_ffxml is True:
                export_ffxml_all = True
            elif isinstance(export_per_residue_ffxml, (list, tuple, set)):
                export_ffxml_targets = {str(x).strip().upper() for x in export_per_residue_ffxml if str(x).strip()}
            else:
                export_ffxml_targets = {str(export_per_residue_ffxml).strip().upper()}

        # Create parameters for each molecule
        def _convert_staged_mol2(residue_name, ligand_charge, skip_charge_computation=False):
            res_upper = residue_name.upper()
            if res_upper not in mol2_conversion_targets:
                return
            mol2_path = os.path.join(par_folder[res_upper], f"{res_upper}.mol2")
            if not os.path.exists(mol2_path):
                return
            if skip_charge_computation:
                # Keep user-provided charges/types; assume mol2 is already prepared.
                return
            temp_output = mol2_path + '.tmp'
            if os.path.exists(temp_output):
                os.remove(temp_output)
            charge_value = ligand_charge if ligand_charge is not None else 0
            command = (
                'antechamber '
                f'-i {shlex.quote(mol2_path)} '
                '-fi mol2 '
                f'-o {shlex.quote(temp_output)} '
                '-fo mol2 '
                '-c bcc '
                f'-nc {charge_value} '
                f'-rn {res_upper} '
                '-s 2 '
                '-pf y\n'
            )
            ret = _run_command(command, self.command_log)
            if ret != 0:
                raise RuntimeError(
                    f'antechamber conversion to GAFF failed for {mol2_path} with exit code {ret}.'
                )
            shutil.move(temp_output, mol2_path)
            mol2_conversion_targets.discard(res_upper)

        for residue in par_folder:

            # Skip metal ions to be processed together with other ligands
            if residue in metal_ligand_values:
                continue

            metal_pdb = None
            metal_charge = None
            if residue in metal_ligand:
                metal_pdb = {}
                metal_charge = {}
                for metal_residue in metal_ligand[residue]:
                    metal_pdb[metal_residue] = '../'+metal_residue+'_parameters/'+metal_residue+'.pdb'
                    metal_charge[metal_residue] = charges[metal_residue]

            if charges and residue in charges:
                charge = charges[residue]
            else:
                charge = 0

            sdf_entry = _get_case_insensitive(ligand_sdf_files or {}, residue)
            if sdf_entry:
                sdf_path = sdf_entry.get("path")
                if not sdf_path or not os.path.exists(sdf_path):
                    raise ValueError(f"SDF file for {residue} not found at {sdf_path}.")
                sdf_index = sdf_entry.get("index", 0)
                charge_method_override = sdf_entry.get("charge_method") or charge_model
                staged_sdf = os.path.join(par_folder[residue], f"{residue}.sdf")
                if not os.path.exists(staged_sdf) or overwrite:
                    _extract_sdf_block(sdf_path, staged_sdf, sdf_index)
                staged_mol2 = os.path.join(par_folder[residue], f"{residue}.mol2")
                if not os.path.exists(staged_mol2) or overwrite:
                    charge_flag = 'rc' if skip_ligand_charge_computation else charge_method_override
                    command = (
                        'antechamber '
                        f'-i {shlex.quote(staged_sdf)} '
                        '-fi mdl '
                        f'-o {shlex.quote(staged_mol2)} '
                        '-fo mol2 '
                        '-pf y '
                        f'-c {charge_flag} '
                        f'-nc {charge} '
                        f'-rn {residue} '
                        '-s 2\n'
                    )
                    ret = _run_command(command, self.command_log)
                    if ret != 0:
                        raise RuntimeError(
                            f'antechamber conversion from SDF failed for {staged_sdf} with exit code {ret}.'
                        )
                staged_path = _stage_external_mol2(residue, staged_mol2, mark_for_conversion=False)
                if staged_path:
                    provided_mol2_residues.add(residue)
                    if residue in provided_frcmod_residues:
                        skip_parameterization_residues.add(residue)
                    _register_mol2_file(residue, staged_path)

            if residue in skip_parameterization_residues:
                continue

            _convert_staged_mol2(residue, charge, skip_ligand_charge_computation)

            if metal_parameters and residue in parameters_folders:
                continue

            print(f'Computing parameters for residue {residue}')
            if residue in metal_ligand:
                print('\tConsidering the followng ions:')
                print(f'\t\t{metal_ligand[residue]}')
            os.chdir(par_folder[residue])

            lig_par = ligandParameters(residue+'.pdb', metal_pdb=metal_pdb, command_log=self.command_log)
            lig_par.getAmberParameters(ligand_charge=charge, overwrite=overwrite,
                                       metal_charge=metal_charge,charge_model=charge_model,
                                       skip_ligand_charge_computation=skip_ligand_charge_computation,
                                       run_acdoctor=run_acdoctor)
            os.chdir('../'*len(par_folder[residue].split('/')))

        # Copy newly generated parameters to the root folder so they can be reused directly.
        for residue, folder in par_folder.items():
            gen_mol2 = os.path.join(folder, f"{residue}.mol2")
            gen_frcmod = os.path.join(folder, f"{residue}.frcmod")
            if os.path.exists(gen_mol2):
                canonical_mol2 = os.path.join(parameters_folder, f"{residue}.mol2")
                shutil.copyfile(gen_mol2, canonical_mol2)
                if not residue_mol2_files[residue]:
                    _register_mol2_file(residue, canonical_mol2)
            if os.path.exists(gen_frcmod):
                dest_frcmod = os.path.join(parameters_folder, f"{residue}.frcmod")
                shutil.copyfile(gen_frcmod, dest_frcmod)
                if not residue_frcmod_files[residue]:
                    _register_frcmod_file(residue, dest_frcmod)
            manifest_path = os.path.join(parameters_folder, f"{residue}.frcmod.list")
            if residue_frcmod_files[residue]:
                try:
                    with open(manifest_path, 'w') as mf:
                        for item in residue_frcmod_files[residue]:
                            entry = os.path.relpath(item, parameters_folder)
                            mf.write(entry + '\n')
                except OSError:
                    pass
            elif os.path.exists(manifest_path):
                try:
                    os.remove(manifest_path)
                except OSError:
                    pass

            mol2_manifest = os.path.join(parameters_folder, f"{residue}.mol2.list")
            if residue_mol2_files[residue]:
                try:
                    with open(mol2_manifest, 'w') as mf:
                        for item in residue_mol2_files[residue]:
                            entry = os.path.relpath(item, parameters_folder)
                            mf.write(entry + '\n')
                except OSError:
                    pass
            elif os.path.exists(mol2_manifest):
                try:
                    os.remove(mol2_manifest)
                except OSError:
                    pass

        # Export per-residue FFXML files when requested (AmberTools path)
        if export_ffxml_all or export_ffxml_targets:
            try:
                import parmed as pmd
            except ImportError:
                raise RuntimeError("ParmEd is required for export_per_residue_ffxml when using the AmberTools backend.")

            # Collect target residues
            targets = set()
            if export_ffxml_all:
                targets.update(par_folder.keys())
                if isinstance(extra_mol2, Mapping):
                    targets.update(str(k).strip().upper() for k in extra_mol2.keys() if str(k).strip())
                if extra_frcmod:
                    if isinstance(extra_frcmod, Mapping):
                        targets.update(str(k).strip().upper() for k in extra_frcmod.keys() if str(k).strip())
            else:
                targets.update({res for res in par_folder.keys() if res.upper() in export_ffxml_targets})
                targets.update(export_ffxml_targets)
            missing = export_ffxml_targets - set(r.upper() for r in par_folder.keys())
            if missing:
                warnings.warn(
                    f"FFXML export requested for residues not parameterized in this model: {sorted(missing)}. "
                    "Attempting export using provided extra_mol2/extra_frcmod entries when available.",
                    RuntimeWarning,
                )

            for residue in sorted(targets):
                # Resolve mol2 and frcmod paths
                mol2_candidates = residue_mol2_files.get(residue) or []
                if not mol2_candidates:
                    mol2_path = os.path.join(parameters_folder, f"{residue}.mol2")
                    if os.path.exists(mol2_path):
                        mol2_candidates = [mol2_path]
                    elif isinstance(extra_mol2, Mapping):
                        # Try to stage extra_mol2 on-demand
                        res_upper = residue.upper()
                        hinted = extra_mol2.get(res_upper) or extra_mol2.get(res_upper.lower()) or extra_mol2.get(res_upper.capitalize())
                        if hinted:
                            try:
                                _, source_path, _ = _resolve_mol2_source(res_upper, hinted)
                                if res_upper in par_folder:
                                    staged = _stage_external_mol2(res_upper, source_path, mark_for_conversion=False)
                                    if staged:
                                        _register_mol2_file(res_upper, staged)
                                        mol2_candidates = [staged]
                                else:
                                    if os.path.exists(source_path):
                                        dest_root = os.path.join(parameters_folder, f"{res_upper}.mol2")
                                        _copyfile_if_needed(source_path, dest_root)
                                        mol2_candidates = [dest_root]
                            except Exception:
                                pass
                frcmod_candidates = residue_frcmod_files.get(residue) or []
                if not frcmod_candidates:
                    frcmod_path = os.path.join(parameters_folder, f"{residue}.frcmod")
                    if os.path.exists(frcmod_path):
                        frcmod_candidates = [frcmod_path]
                    elif extra_frcmod:
                        res_upper = residue.upper()
                        try:
                            # Reuse resolver to locate frcmod from user inputs
                            source_entry = None
                            if isinstance(extra_frcmod, Mapping):
                                source_entry = extra_frcmod.get(res_upper) or extra_frcmod.get(res_upper.lower()) or extra_frcmod.get(res_upper.capitalize())
                            elif isinstance(extra_frcmod, (list, tuple)):
                                source_entry = extra_frcmod
                            if source_entry is not None:
                                if isinstance(source_entry, (list, tuple)):
                                    source_entry = source_entry[0]
                                res_hint, source_path = _resolve_frcmod_source(res_upper, source_entry)
                                if source_path:
                                    dest_root = _copy_into_par_folder(res_hint, source_path) or source_path
                                    if res_hint not in par_folder:
                                        dest_root = os.path.join(parameters_folder, f"{res_hint}.frcmod")
                                        _copyfile_if_needed(source_path, dest_root)
                                    _register_frcmod_file(res_hint, dest_root)
                                    frcmod_candidates = [dest_root]
                        except Exception:
                            pass

                if not mol2_candidates:
                    warnings.warn(f"Skipping FFXML export for {residue}: no mol2 found.", RuntimeWarning)
                    continue
                if not frcmod_candidates:
                    warnings.warn(f"Skipping FFXML export for {residue}: no frcmod found.", RuntimeWarning)
                    continue

                # Build a temporary tleap script to generate amber files for the residue alone
                tmp_prmtop = os.path.join(parameters_folder, f"{residue}_ffxml.prmtop")
                tmp_inpcrd = os.path.join(parameters_folder, f"{residue}_ffxml.inpcrd")
                tlin = os.path.join(parameters_folder, f"{residue}_ffxml.leap")
                with open(tlin, "w") as tl:
                    tl.write("source leaprc.protein.ff14SB\n")
                    tl.write("source leaprc.gaff\n")
                    tl.write(f'loadamberparams "{frcmod_candidates[0]}"\n')
                    tl.write(f'{residue} = loadmol2 "{mol2_candidates[0]}"\n')
                    tl.write(f'saveamberparm {residue} "{tmp_prmtop}" "{tmp_inpcrd}"\n')
                    tl.write("quit\n")
                ret = _run_command(f"tleap -s -f \"{tlin}\"", self.command_log)
                if ret != 0 or (not os.path.exists(tmp_prmtop) or not os.path.exists(tmp_inpcrd)):
                    warnings.warn(f"tleap failed while generating FFXML inputs for {residue}", RuntimeWarning)
                    continue

                try:
                    struct = pmd.load_file(tmp_prmtop, tmp_inpcrd)
                    param_set = pmd.openmm.parameters.OpenMMParameterSet.from_structure(struct)
                    from parmed.modeller.residue import ResidueTemplate
                    for res in struct.residues:
                        tmpl = ResidueTemplate.from_residue(res)
                        # keep first occurrence per name
                        if tmpl.name not in param_set.residues:
                            param_set.residues[tmpl.name] = tmpl
                    ffxml_path = os.path.join(parameters_folder, f"{residue}.ffxml")
                    param_set.write(ffxml_path)
                except Exception as exc:
                    warnings.warn(f"Failed to save FFXML for {residue}: {exc}", RuntimeWarning)
                finally:
                    for path in (tmp_prmtop, tmp_inpcrd, tlin):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

        # Renumber PDB
        renum_pdb = pdb_file.replace('.pdb', '_renum.pdb')
        if not os.path.exists(renum_pdb) or regenerate_amber_files:
            command =  'pdb4amber -i '
            command += pdb_file+' '
            command += '-o '+renum_pdb+'\n'
            _run_command(command, self.command_log)

        # Parameterize metal complex with MCPB.py
        mcpb_input_spec = None
        if use_mcpb:

            # Copy frcmmod file from previous optimization
            if metal_parameters:
                _copyfile_if_needed(parameters_folders['mcpbpy.frcmod'],
                                    parameters_folder+'/'+self.pdb_name+'_mcpbpy.frcmod')

            if use_site_driven_mcpb:
                mcpb_input_spec = _build_site_driven_mcpb_inputs(
                    resolved_mcpb_config,
                    par_folder,
                    residue_mol2_files,
                    residue_frcmod_files,
                    parameters_folder,
                    self.pdb_name,
                )
            else:
                # Get ion ID
                metal_pdb = PDBFile(renum_pdb)
                ion_ids = {}
                for residue in metal_pdb.topology.residues():
                    for r in metal_ligand:
                        if residue.name in metal_ligand[r]:
                            atoms = list(residue.atoms())
                            if len(atoms) > 1:
                                raise ValueError('Ion residue should contain only one atom!')
                            ion_ids[residue.name] = atoms[0].index+1

            # Copy input files
            residues_to_stage = list(par_folder.keys())
            if mcpb_input_spec:
                residues_to_stage = list(mcpb_input_spec["staged_residue_names"])

            for residue in residues_to_stage:
                residue = residue.upper()

                if residue in metal_ligand_values:
                    continue

                if metal_parameters and residue in parameters_folders:
                    continue

                if residue not in par_folder:
                    continue

                # Copy metal ions files
                if residue in metal_ligand:
                    for m in metal_ligand[residue]:
                        _copyfile_if_needed(par_folder[residue]+'/'+m+'.mol2',
                                            parameters_folder+'/'+m+'.mol2')

                mol2_source = _resolve_residue_parameter_file(
                    residue,
                    residue_mol2_files,
                    par_folder,
                    parameters_folder,
                    '.mol2',
                )
                if mol2_source is not None:
                    _copyfile_if_needed(mol2_source, os.path.join(parameters_folder, f'{residue}.mol2'))

                frcmod_source = _resolve_residue_parameter_file(
                    residue,
                    residue_frcmod_files,
                    par_folder,
                    parameters_folder,
                    '.frcmod',
                )
                if frcmod_source is not None:
                    canonical_path = os.path.join(parameters_folder, f'{residue}.frcmod')
                    _copyfile_if_needed(frcmod_source, canonical_path)
                    if canonical_path not in residue_frcmod_files.get(residue, []):
                        _register_frcmod_file(residue, canonical_path)

            if not metal_parameters:

                input_file = pdb_file.replace('.pdb', '.in')
                with open(input_file, 'w') as f:
                    f.write('original_pdb '+self.pdb_name+'_renum.pdb'+'\n')
                    group_name = self.pdb_name
                    cut_off = 2.8
                    ion_ids_list = []
                    ion_mol2_files = []
                    naa_mol2_files = []
                    frcmod_files = []
                    add_bonded_pairs = []
                    if mcpb_input_spec:
                        group_name = mcpb_input_spec["group_name"]
                        cut_off = mcpb_input_spec["cut_off"]
                        ion_ids_list = list(mcpb_input_spec["ion_ids"])
                        ion_mol2_files = list(mcpb_input_spec["ion_mol2files"])
                        naa_mol2_files = list(mcpb_input_spec["naa_mol2files"])
                        frcmod_files = list(mcpb_input_spec["frcmod_files"])
                        add_bonded_pairs = list(mcpb_input_spec["add_bonded_pairs"])
                    else:
                        ion_ids_list = []
                        for residue in par_folder:
                            if residue in metal_ligand:
                                for m in metal_ligand[residue]:
                                    ion_ids_list.append(ion_ids[m])
                                    ion_mol2_files.append(f'{m}.mol2')
                            if residue in metal_ligand_values:
                                continue
                            naa_mol2_files.append(f'{residue}.mol2')
                            entries = residue_frcmod_files.get(residue, [])
                            if entries:
                                frcmod_file = os.path.basename(entries[0])
                            else:
                                frcmod_file = f'{residue}.frcmod'
                            frcmod_files.append(frcmod_file)

                    f.write('group_name '+group_name+'\n')
                    f.write('software_version g09\n')
                    f.write('force_field '+force_field+'\n')
                    f.write(f'cut_off {cut_off}\n')
                    ion_ids_line = 'ion_ids'
                    for ion_id in ion_ids_list:
                        ion_ids_line += ' '+str(ion_id)
                    f.write(ion_ids_line+'\n')
                    if ion_mol2_files:
                        ion_mol2 = 'ion_mol2files'
                        for ion_mol2_file in ion_mol2_files:
                            ion_mol2 += ' '+ion_mol2_file
                        f.write(ion_mol2+'\n')
                    if naa_mol2_files:
                        naa_mol2 = 'naa_mol2files'
                        for naa_mol2_file in naa_mol2_files:
                            naa_mol2 += ' '+naa_mol2_file
                        f.write(naa_mol2+'\n')
                    if frcmod_files:
                        frcmod = 'frcmod_files'
                        for frcmod_file in frcmod_files:
                            frcmod += ' '+frcmod_file
                        f.write(frcmod+'\n')
                    if add_bonded_pairs:
                        add_bonded_pairs_line = 'add_bonded_pairs'
                        for atom_1, atom_2 in add_bonded_pairs:
                            add_bonded_pairs_line += f' {atom_1}-{atom_2}'
                        f.write(add_bonded_pairs_line+'\n')

                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 1\n'
                _run_command(command, self.command_log)

                # Set gaussian calculations for small
                commands = []

                command = ''
                finished = _checkIfGaussianFinished(self.pdb_name+'_small_opt.log')
                if not finished:
                    command = 'cd '+parameters_folder+'\n'
                    _changeGaussianCPUS(self.pdb_name+'_small_opt.com', cpus)
                    command +=  'g09 '+self.pdb_name+'_small_opt.com > '+self.pdb_name+'_small_opt.log\n'

                finished = _checkIfGaussianFinished(self.pdb_name+'_small_fc.log')
                if not finished:
                    _changeGaussianCPUS(self.pdb_name+'_small_fc.com', cpus)
                    command += 'g09 '+self.pdb_name+'_small_fc.com > '+self.pdb_name+'_small_fc.log\n'
                    command += 'formchk '+self.pdb_name+'_small_opt.chk > '+self.pdb_name+'_small_opt.fchk\n'
                if command != '':
                    command += 'cd ../\n'
                    commands.append(command)

                # Set gaussian calculations for large
                command = ''
                finished = _checkIfGaussianFinished(self.pdb_name+'_large_mk.log')
                if not finished:
                    command = 'cd '+parameters_folder+'\n'
                    _changeGaussianCPUS(self.pdb_name+'_large_mk.com', cpus)
                    command += 'g09 '+self.pdb_name+'_large_mk.com > '+self.pdb_name+'_large_mk.log\n'
                if command != '':
                    command += 'cd ../\n'
                    commands.append(command)

                os.chdir('../'*len(parameters_folder.split('/')))
                if commands:# and not metal_parameters:
                    if return_qm_jobs:
                        print('Returning QM jobs')
                        return commands
                    else:
                        print('Computing QM parameters')
                        for command in commands:
                            _run_command(command, self.command_log)

                print('QM calculations finished.')

                # Run step 2 of the MCPB protocol
                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 2\n'
                _run_command(command, self.command_log)
                os.chdir('../'*len(parameters_folder.split('/')))

                # Run step 3 of the MCPB protocol
                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 3\n'
                _run_command(command, self.command_log)
                os.chdir('../'*len(parameters_folder.split('/')))

                # Run step 4 of the MCPB protocol
                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 4\n'
                _run_command(command, self.command_log)
                os.chdir('../'*len(parameters_folder.split('/')))

        if not build_full_system:
            return

        # Generate set of metal ligand values
        metal_ligand_values = []
        for r in metal_ligand:
            if isinstance(metal_ligand[r], str): # Convert into a list
                metal_ligand[r] = [metal_ligand[r]]
            for m in metal_ligand[r]:
                metal_ligand_values.append(m)

        # Create tleap input file
        with open(parameters_folder+'/tleap.in', 'w') as tlf:
            mol2_loaded_residues = set()
            tlf.write('source leaprc.protein.ff14SB\n')
            tlf.write('source leaprc.gaff\n')
            tlf.write('source leaprc.water.tip3p\n')
            if membrane_system:
                for tleap_source in membrane_system["tleap_sources"]:
                    tlf.write(f'source {tleap_source}\n')
            if extra_force_field:
                tlf.write('source '+extra_force_field_source+'\n')

            if use_mcpb:

                mcpb_site_residue_names = []
                if mcpb_input_spec:
                    mcpb_site_residue_names = [residue.upper() for residue in mcpb_input_spec["site_residue_names"]]
                else:
                    for residue in par_folder:
                        if residue in metal_ligand_values:
                            continue
                        mcpb_site_residue_names.append(residue.upper())
                mcpb_site_residue_names = _ordered_unique(mcpb_site_residue_names)

                # Generate frcmod files
                if metal_parameters:
                    mcpb_pdb = renum_pdb
                else:
                    mcpb_pdb = parameters_folder+'/'+self.pdb_name+'_mcpbpy.pdb'
                pdb = PDBFile(mcpb_pdb)

                # Get mapping as tuples for residues
                missing_atoms = []
                for residue in getNonProteinResidues(pdb.topology, skip_residues=skip_ligands):
                    residue_name = residue.name.upper()
                    if residue_name not in mcpb_site_residue_names:
                        continue
                    mol2_path = os.path.join(parameters_folder, residue_name+'.mol2')
                    if os.path.exists(mol2_path):
                        missing_atoms += getMissingAtomTypes(mol2_path)

                mcpb_frcmod = parameters_folder+'/'+self.pdb_name+'_mcpbpy.frcmod'
                atom_types = getAtomTypes(mcpb_frcmod, missing_atoms)
                tlf.write('addAtomTypes {\n')
                for atom in atom_types:
                    tlf.write('\t{ "'+atom+'"  "'+atom_types[atom]+'" "sp3" }\n')
                tlf.write('}\n')

                for residue in getNonProteinResidues(pdb.topology, skip_residues=skip_ligands):
                    tlf.write(residue.name+' = loadmol2 '+parameters_folder+'/'+residue.name+'.mol2\n')

            for res_upper, mol2_paths in residue_mol2_files.items():
                if not mol2_paths:
                    continue
                primary_mol2 = mol2_paths[0]
                if os.path.exists(primary_mol2) and res_upper not in mol2_loaded_residues:
                    tlf.write(f'{res_upper} = loadmol2 "{primary_mol2}"\n')
                    mol2_loaded_residues.add(res_upper)

            frcmod_residue_order = list(par_folder.keys())
            for residue_name in residue_frcmod_files:
                if residue_name not in par_folder:
                    frcmod_residue_order.append(residue_name)

            for residue in frcmod_residue_order:
                frcmod_paths = residue_frcmod_files.get(residue) or []
                if frcmod_paths:
                    for frcmod_path in frcmod_paths:
                        if os.path.exists(frcmod_path):
                            tlf.write(f'loadamberparams {frcmod_path}\n')
                    if residue not in par_folder:
                        continue
                elif residue not in par_folder:
                    continue

                if residue in metal_ligand_values:
                    continue
                prepi_path = os.path.join(par_folder[residue], f'{residue}.prepi')
                ffxml_marker = os.path.join(par_folder[residue], f".ffxml_generated_{residue.lower()}")
                ffxml_marker_root = os.path.join(parameters_folder, f".ffxml_generated_{residue.lower()}")
                ffxml_generated = os.path.exists(ffxml_marker) or os.path.exists(ffxml_marker_root)
                if not use_mcpb:
                    if os.path.exists(prepi_path) and not ffxml_generated:
                        tlf.write(f'loadamberprep {prepi_path}\n')
                    else:
                        mol2_paths = residue_mol2_files.get(residue, [])
                        if mol2_paths:
                            primary_mol2 = mol2_paths[0]
                            if os.path.exists(primary_mol2) and residue not in mol2_loaded_residues:
                                tlf.write(f'{residue} = loadmol2 "{primary_mol2}"\n')
                                mol2_loaded_residues.add(residue)
                        else:
                            mol2_path = os.path.join(parameters_folder, f'{residue}.mol2')
                            if os.path.exists(mol2_path) and residue not in mol2_loaded_residues:
                                tlf.write(f'{residue} = loadmol2 "{mol2_path}"\n')
                                mol2_loaded_residues.add(residue)
                if not frcmod_paths:
                    fallback_frcmod = os.path.join(par_folder[residue], f'{residue}.frcmod')
                    if os.path.exists(fallback_frcmod):
                        tlf.write(f'loadamberparams {fallback_frcmod}\n')
                    else:
                        fallback_root = os.path.join(parameters_folder, f'{residue}.frcmod')
                        if os.path.exists(fallback_root):
                            tlf.write(f'loadamberparams {fallback_root}\n')

            if use_mcpb:
                tlf.write('loadamberparams '+mcpb_frcmod+'\n')
                tlf.write('mol = loadpdb '+mcpb_pdb+'\n')
            else:
                tlf.write('mol = loadpdb '+renum_pdb+'\n')

            if non_standard_residues:
                if not add_bonds:
                    add_bonds = []

                for residue in non_standard_residues:
                    residues = getResiduesByName(self.modeller.topology, residue)
                    for r in residues:
                        tlf.write('set mol.'+r.id+' connect0 mol.'+r.id+'.N\n')
                        tlf.write('set mol.'+r.id+' connect1 mol.'+r.id+'.C\n')
                        # cn_bond = ((r.chain.id, int(r.id)-1, 'C'),(r.chain.id, int(r.id), 'N'))
                        # nc_bond = ((r.chain.id, int(r.id), 'C'),(r.chain.id, int(r.id)+1, 'N'))
                        # add_bonds.append(cn_bond)
                        # add_bonds.append(nc_bond)

            # Add bonds
            if add_bonds:

                input_pdb_object = PDBFile(self.input_pdb)
                input_positions = np.array(
                    [vec.value_in_unit(nanometer) for vec in input_pdb_object.positions], dtype=float
                )
                saved_pdb_object = PDBFile(pdb_file)
                saved_positions = np.array(
                    [vec.value_in_unit(nanometer) for vec in saved_pdb_object.positions], dtype=float
                )
                renum_pdb_object = PDBFile(renum_pdb)

                saved_residue_map = {}
                for chain in saved_pdb_object.topology.chains():
                    for residue in chain.residues():
                        saved_residue_map[(chain.id, int(residue.id))] = residue

                saved_atom_lookup = {}
                saved_atoms_by_signature = defaultdict(list)
                for atom in saved_pdb_object.topology.atoms():
                    key = (atom.residue.chain.id, int(atom.residue.id), atom.name)
                    saved_atom_lookup[key] = atom
                    saved_atoms_by_signature[(atom.residue.name, atom.name)].append(atom)

                input_residues = {}
                input_atoms = {}
                for atom in input_pdb_object.topology.atoms():
                    chain_id = atom.residue.chain.id
                    res_id = int(atom.residue.id)
                    key = (chain_id, res_id)
                    input_residues[key] = atom.residue
                    input_atoms[(chain_id, res_id, atom.name)] = atom

                saved_to_renum = {}
                for saved_residue, renum_residue in zip(
                    saved_pdb_object.topology.residues(), renum_pdb_object.topology.residues()
                ):
                    saved_key = (saved_residue.chain.id, int(saved_residue.id))
                    saved_to_renum[saved_key] = (renum_residue.chain.id, int(renum_residue.id))

                def _locate_atom(atom_spec):
                    chain_id, res_id, atom_name = atom_spec
                    direct_key = (chain_id, res_id, atom_name)
                    atom = saved_atom_lookup.get(direct_key)
                    if atom is not None:
                        saved_chain = atom.residue.chain.id
                        saved_res = int(atom.residue.id)
                    else:
                        input_atom = input_atoms.get(direct_key)
                        if input_atom is None:
                            raise ValueError(
                                f'Atom {atom_name} in residue {chain_id}:{res_id} was not found in the input structure.'
                            )
                        input_residue = input_residues[(chain_id, res_id)]
                        candidates = saved_atoms_by_signature.get((input_residue.name, atom_name), [])
                        target_pos = input_positions[input_atom.index]
                        saved_chain = None
                        saved_res = None
                        for candidate in candidates:
                            cand_pos = saved_positions[candidate.index]
                            if np.linalg.norm(cand_pos - target_pos) < 1e-4:
                                saved_chain = candidate.residue.chain.id
                                saved_res = int(candidate.residue.id)
                                atom = candidate
                                break
                        if saved_chain is None:
                            raise ValueError(
                                f'Unable to map atom {atom_name} from residue {chain_id}:{res_id} '
                                'to the renumbered structure.'
                            )
                    renum_chain, renum_res = saved_to_renum.get((saved_chain, saved_res), (saved_chain, saved_res))
                    return renum_chain, renum_res, atom_name

                for bond in add_bonds:
                    atom1 = _locate_atom(bond[0])
                    atom2 = _locate_atom(bond[1])
                    tlf.write(
                        f'bond mol.{atom1[1]}.{atom1[2]} '
                        f'mol.{atom2[1]}.{atom2[2]}\n'
                    )

            if solvate:
                buffer_expr = formatSolvateboxBuffer(solvatebox_buffer)
                solvate_command = f"solvatebox mol TIP3PBOX {buffer_expr}"
                if bool(solvatebox_iso) and isinstance(solvatebox_buffer, (int, float)):
                    solvate_command += " iso"
                tlf.write(solvate_command + '\n')

            #Add ions with addIons2 (fast) or addIonsRand (slow)
            if add_counterionsRand:
                add_counterions = False
            if add_counterions:
                tlf.write('addIons2 mol Na+ 0\n')
                tlf.write('addIons2 mol Cl- 0\n')
            if add_counterionsRand:
                tlf.write('addIonsRand mol Na+ 0\n')
                tlf.write('addIonsRand mol Cl- 0\n')


            if save_amber_pdb:
                tlf.write('savepdb mol '+parameters_folder+'/'+self.pdb_name+'_amber.pdb\n')

            tlf.write('saveamberparm mol '+parameters_folder+'/'+self.pdb_name+'.prmtop '+parameters_folder+'/'+self.pdb_name+'.inpcrd\n')

        _run_command('tleap -s -f '+parameters_folder+'/tleap.in', self.command_log)

        # Define prmtop and inpcrd file paths
        self.prmtop_file = parameters_folder+'/'+self.pdb_name+'.prmtop'
        self.inpcrd_file = parameters_folder+'/'+self.pdb_name+'.inpcrd'

        # Set topology and positions to amber's
        prmtop = AmberPrmtopFile(self.prmtop_file)
        inpcrd = AmberInpcrdFile(self.inpcrd_file)

        self.modeller.topology = prmtop.topology
        self.modeller.positions = inpcrd.positions
        self._refresh_positions_cache()

    def savePDB(self, output_file):
        with open(output_file, 'w') as of:
            PDBFile.writeFile(self.modeller.topology, self.modeller.positions, of)

    def centerPositionsAtCOM(self, output_unit=nanometer):
        masses = np.array([a.element.mass.value_in_unit(amu) for a in self.modeller.topology.atoms()])
        masses = np.repeat(masses, 3, axis=0).reshape((masses.shape[0], 3))
        com = np.sum(self.positions*masses, axis=0)/np.sum(masses[:,0])
        self.positions = self.positions-com
        self.modeller.positions = _getPositionsArrayAsVector(self.positions)

    def get_command_log(self):
        """Return a copy of the recorded external command log."""
        return list(self.command_log)

class ligandParameters:

    def __init__(self, ligand_pdb, metal_pdb=None, command_log=None):

        self.ligand_pdb = ligand_pdb
        self.pdb = PDBFile(self.ligand_pdb)
        self.pdb_name = ligand_pdb.split('/')[-1].replace('.pdb', '')
        self.metal_pdb = metal_pdb
        self.metal = bool(metal_pdb)
        self.command_log = command_log

        res_count = 0
        for residue in self.pdb.topology.residues():
            res_count += 1
            self.resname = residue.name

        if res_count != 1:
            raise ValueError('A PDB with a single residue must be given for parameterization!')

    def getAmberParameters(self, charge_model='bcc', ligand_charge=0, metal_charge=None, overwrite=False,
                           skip_ligand_charge_computation=False, run_acdoctor=True):
        def _run_acdoctor_on_mol2(mol2_file):
            marker_path = f".ffxml_generated_{self.resname.lower()}"
            if os.path.exists(marker_path) or os.path.exists(os.path.join(os.getcwd(), marker_path)):
                return
            if not os.path.exists(mol2_file):
                raise FileNotFoundError(f'acdoctor input {mol2_file} not found.')
            command = (
                'antechamber '
                f'-i {shlex.quote(mol2_file)} '
                '-fi mol2 '
                '-o /dev/null '
                '-fo mol2 '
                '-j 0 '
                '-dr y '
                '-s 2 '
                '-pf y\n'
            )
            ret_code = _run_command(command, self.command_log)
            if ret_code != 0:
                raise RuntimeError(f'acdoctor (antechamber) failed for {mol2_file} with exit code {ret_code}.')

        # Execute pdb4amber to generate a renumbered PDB
        if not os.path.exists(self.resname+'_renum.pdb') or overwrite:
            command  = 'pdb4amber '
            command += '-i '+self.ligand_pdb+' '
            command += '-o '+self.resname+'_renum.pdb '
            _run_command(command, self.command_log)

        # Run antechamber to create a mol2 file with the atomic charges
        mol2_path = self.resname+'.mol2'
        mol2_generated = False
        if not skip_ligand_charge_computation or overwrite:
            if not os.path.exists(mol2_path) or overwrite:
                command = 'antechamber '
                command += '-i '+self.resname+'_renum.pdb '
                command += '-fi pdb '
                command += '-o '+self.resname+'.mol2 '
                command += '-fo mol2 '
                command += '-pf y '
                command += '-c '+charge_model+' '
                command += '-nc '+str(ligand_charge)+' '
                command += '-s 2\n'
                _run_command(command, self.command_log)
                mol2_generated = True
        elif not os.path.exists(mol2_path):
            raise FileNotFoundError(
                f"skip_ligand_charge_computation=True but {mol2_path} is missing; "
                "provide an existing mol2 with charges or disable the skip flag."
            )

        # Run antechamber to create a prepi file with the atomic charges when allowed
        prepi_path = self.resname+'.prepi'
        if not skip_ligand_charge_computation or overwrite:
            if not os.path.exists(prepi_path) or overwrite:
                command = 'antechamber '
                command += '-i '+self.resname+'_renum.pdb '
                command += '-fi pdb '
                command += '-o '+prepi_path+' '
                command += '-fo prepi '
                command += '-pf y '
                command += '-c '+charge_model+' '
                command += '-nc '+str(ligand_charge)+' '
                command += '-s 2\n'
                _run_command(command, self.command_log)

        # Run parmchk to check which forcefield parameters will be used
        frcmod_path = self.resname+'.frcmod'
        need_parmchk = overwrite or not os.path.exists(frcmod_path)
        if need_parmchk and not mol2_generated and run_acdoctor:
            _run_acdoctor_on_mol2(mol2_path)
        if need_parmchk:
            command  = 'parmchk2 '
            command += '-i '+mol2_path+' '
            command += '-o '+frcmod_path+' '
            command += '-f mol2\n'
            _run_command(command, self.command_log)

        # Parameterize metal pdb if given
        if self.metal_pdb:
            for m in self.metal_pdb:
                metal_mol2 = self.metal_pdb[m].split('/')[-1].replace('.pdb','.mol2')

                if not os.path.exists(metal_mol2) or overwrite:

                    if not metal_charge:
                        raise ValueError('You must give the metal_charge if using a ligand with metal!')
                    self.metal_charge = metal_charge
                    command  = 'metalpdb2mol2.py '
                    command += '-i '+self.metal_pdb[m]+' '
                    command += '-o '+metal_mol2+' '
                    command += '-c '+str(self.metal_charge[m])+'\n'
                    _run_command(command, self.command_log)

def _getPositionsArrayAsVector(positions):
    _ensure_openmm("_getPositionsArrayAsVector")
    v3_positions = []
    for p in positions:
        v3_positions.append(quantity.Quantity(Vec3(*p), unit=nanometer))
    return v3_positions

def _copyFFFiles(output_folder, ff):
    """
    Copy all forcefield files from the specified folder in the prepare_proteins package.

    Parameters
    ==========
    output_folder : str
        Path to the folder where the forcefield files will be copied.
    ff : str
        Name of the forcefield folder to copy files from.
    """
    ffs = ['parmBSC1', 'parmBSC2']
    if ff not in ffs:
        raise ValueError(f"{ff} not implemented!")

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the forcefield folder and copy them to the output folder
    for filename in resource_listdir(
        "prepare_proteins", f"prepare_proteins/MD/ff_files/{ff}"
    ):
        if filename == '__init__.py':
            continue

        destination_file = os.path.join(output_folder, filename)

        # Give ff path to leaprc file
        if 'leaprc.bsc' in destination_file:
            with resource_stream(
                "prepare_proteins",
                f"prepare_proteins/MD/ff_files/{ff}/{filename}",
            ) as stream:
                with io.TextIOWrapper(stream) as source_file:
                    with open(destination_file, 'w') as dest_handle:
                        for line in source_file:
                            line = line.replace('FF_PATH', output_folder)
                            dest_handle.write(line)
        else:
            with resource_stream(
                "prepare_proteins",
                f"prepare_proteins/MD/ff_files/{ff}/{filename}",
            ) as source_file:
                with open(destination_file, 'wb') as dest_handle:
                    shutil.copyfileobj(source_file, dest_handle)
