from __future__ import annotations

import fileinput
import gc
import io
import importlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
import copy
import re
import pkg_resources
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

# --- Debug printing helper (no-op if debug=False) ---
def _dbg(debug: bool, msg: str = "", *args):
    if debug:
        try:
            print(msg.format(*args))
        except Exception:
            # Be resilient to bad formats/objects
            print(msg)

def _mask_runs(mask):
    """Return list of (start, end, value) for contiguous runs in a boolean mask."""
    runs = []
    if not mask:
        return runs
    start = 0
    cur = mask[0]
    for i, v in enumerate(mask[1:], start=1):
        if v != cur:
            runs.append((start, i - 1, cur))
            start = i
            cur = v
    runs.append((start, len(mask) - 1, cur))
    return runs

def _longest_true_run(bool_list):
    """
    Return (start_idx, end_idx) of the longest contiguous True run in bool_list.
    If no True values, return None.
    """
    best_len = 0
    best = None
    cur_len = 0
    cur_start = 0
    for i, v in enumerate(bool_list):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best = (cur_start, i)
        else:
            cur_len = 0
    return best

def _term_anchored_core(mapping_pairs, chain_idx_list):
    """
    Core bounded by the reference termini:
      - mapping_pairs: list of (ref_idx, tgt_idx)
      - chain_idx_list: global target indices for this chain
    Return (core_start, core_end) in GLOBAL target indices using the target residues
    aligned to the MIN and MAX ref_idx present for this chain. If only one anchor is
    present, fall back to the earliest/latest target-aligned residue.
    """
    chain_set = set(chain_idx_list)
    chain_pairs = [(r, t) for (r, t) in mapping_pairs if t in chain_set]
    if not chain_pairs:
        return None

    # Anchor strictly by reference indices (termini)
    r_min, t_min = min(chain_pairs, key=lambda x: x[0])
    r_max, t_max = max(chain_pairs, key=lambda x: x[0])

    # If both anchors exist (typical), use them
    if r_min != r_max:
        core_start = min(t_min, t_max)
        core_end = max(t_min, t_max)
        return (core_start, core_end)

    # Fallback (rare): only one ref anchor effectively present → use first/last target
    first_tgt = min(chain_pairs, key=lambda x: x[1])[1]
    last_tgt = max(chain_pairs, key=lambda x: x[1])[1]
    core_start = min(first_tgt, last_tgt)
    core_end = max(first_tgt, last_tgt)
    return (core_start, core_end)

def _wrap_pyrosetta_command(command: str, pyrosetta_env: Optional[str]) -> str:
    """
    Wrap the command in a conda environment activation if `pyrosetta_env` is given.
    """
    if not pyrosetta_env:
        return command

    try:
        result = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Cannot find conda base path. Ensure `conda` is installed and on PATH."
        ) from exc

    conda_base = result.stdout.strip()
    conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
    if not os.path.exists(conda_sh):
        raise FileNotFoundError(
            f"Cannot locate '{conda_sh}'. Run `conda init` or install Conda."
        )

    conda_prefix = (
        f"source {conda_sh} >/dev/null 2>&1 && "
        f"conda activate {pyrosetta_env} >/dev/null 2>&1 &&"
    )

    return f"bash -lc \"{conda_prefix} {command.strip()}\""


def _sync_ligand_params_to_shared_folder(docking_folder: str):
    """
    Ensure Rosetta sees the generated ligand params by mirroring them into docking_folder/params.
    """
    ligand_params_root = os.path.join(docking_folder, "ligand_params")
    if not os.path.isdir(ligand_params_root):
        return

    shared_params = os.path.join(docking_folder, "params")
    os.makedirs(shared_params, exist_ok=True)

    for ligand in os.listdir(ligand_params_root):
        ligand_dir = os.path.join(ligand_params_root, ligand)
        if not os.path.isdir(ligand_dir):
            continue
        for suffix in (".params", ".pdb", "_conformers.pdb"):
            src = os.path.join(ligand_dir, f"{ligand}{suffix}")
            if not os.path.exists(src):
                continue
            dst = os.path.join(shared_params, f"{ligand}{suffix}")
            shutil.copyfile(src, dst)


def _parse_ligand_atom_map(
    docking_folder: str,
    ligand: str,
    chain_override: Optional[str] = None,
    resseq_override: Optional[int] = None,
) -> Dict[str, Tuple[str, int, str]]:
    ligand_pdb = os.path.join(docking_folder, "ligand_params", ligand, f"{ligand}.pdb")
    if not os.path.exists(ligand_pdb):
        raise FileNotFoundError(f"Ligand PDB not found: {ligand_pdb}")

    atom_map: Dict[str, Tuple[str, int, str]] = {}
    with open(ligand_pdb) as lf:
        for line in lf:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if not atom_name or atom_name in atom_map:
                continue
            chain = line[21].strip()
            if not chain:
                chain = " "
            resseq_str = line[22:26].strip()
            if not resseq_str.isdigit():
                raise ValueError(f"Unexpected residue number '{resseq_str}' in {ligand_pdb}")
            resseq = int(resseq_str)
            if chain_override is not None:
                chain = chain_override
            if resseq_override is not None:
                resseq = resseq_override
            atom_map[atom_name] = (chain, resseq, atom_name)
    if not atom_map:
        raise ValueError(f"No atoms parsed from ligand PDB: {ligand_pdb}")
    return atom_map


def _list_ligand_atom_names(docking_folder: str, ligand: str) -> List[str]:
    pdb_path = os.path.join(docking_folder, "ligand_params", ligand, f"{ligand}.pdb")
    if not os.path.exists(pdb_path):
        return []

    atom_names = []
    with open(pdb_path) as lf:
        for line in lf:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name and atom_name not in atom_names:
                atom_names.append(atom_name)
    return atom_names


def _list_params_atom_names(docking_folder: str, ligand: str) -> Tuple[List[str], str]:
    params_path = os.path.join(docking_folder, "params", f"{ligand}.params")
    if not os.path.exists(params_path):
        return [], params_path

    atom_names = []
    with open(params_path) as pf:
        for line in pf:
            stripped = line.strip()
            if not stripped or not stripped.startswith("ATOM"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            atom_names.append(parts[1])
    return atom_names, params_path


def _list_pdb_chains(pdb_path: str) -> Set[str]:
    chains = set()
    if not os.path.exists(pdb_path):
        return chains
    with open(pdb_path) as pf:
        for line in pf:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain = line[21].strip()
            if chain:
                chains.add(chain)
    return chains


def _parse_res_num_line(out_path: str) -> Dict[str, Tuple[int, int]]:
    if not os.path.exists(out_path):
        return {}
    with open(out_path) as of:
        for line in of:
            if not line.startswith("RES_NUM"):
                continue
            tokens = line.strip().split()
            if len(tokens) < 3:
                continue
            ranges: Dict[str, Tuple[int, int]] = {}
            for token in tokens[1:-1]:
                if ":" not in token:
                    continue
                chain, range_str = token.split(":", 1)
                if "-" in range_str:
                    start, end = range_str.split("-", 1)
                else:
                    start = range_str
                    end = range_str
                try:
                    ranges[chain] = (int(start), int(end))
                except ValueError:
                    continue
            return ranges
    return {}


def _collect_ligand_chain_overrides(
    docking_folder: str, model_ligands, separator: str
) -> Dict[str, Tuple[str, int]]:
    overrides: Dict[str, Tuple[str, int]] = {}
    output_models_path = os.path.join(docking_folder, "output_models")
    input_models_path = os.path.join(docking_folder, "input_models")
    for model_ligand in model_ligands:
        if separator not in model_ligand:
            continue
        base_model, _ = model_ligand.split(separator, 1)
        base_pdb = os.path.join(input_models_path, f"{base_model}.pdb")
        base_chains = _list_pdb_chains(base_pdb)
        out_path = os.path.join(output_models_path, model_ligand, f"{model_ligand}.out")
        if not os.path.exists(out_path):
            sc_path = os.path.join(output_models_path, model_ligand, f"{model_ligand}.sc")
            out_path = sc_path if os.path.exists(sc_path) else None
        if out_path is None:
            continue
        chain_ranges = _parse_res_num_line(out_path)
        if not chain_ranges:
            continue
        ligand_chains = [chain for chain in chain_ranges if chain not in base_chains]
        if not ligand_chains:
            ligand_chains = list(chain_ranges.keys())
        if not ligand_chains:
            continue
        chain = ligand_chains[0]
        res_start, _ = chain_ranges.get(chain, (None, None))
        if res_start is None:
            continue
        overrides[model_ligand] = (chain, res_start)
    return overrides


def _collect_requested_model_ligand_pairs(atom_pairs) -> Set[Tuple[str, str]]:
    entries = set()
    if not isinstance(atom_pairs, dict):
        return entries
    for model, ligands in atom_pairs.items():
        if not isinstance(ligands, dict):
            continue
        for ligand in ligands:
            if isinstance(ligand, str):
                entries.add((model, ligand))
    return entries


def _collect_requested_ligands(atom_pairs) -> Set[str]:
    return {ligand for _, ligand in _collect_requested_model_ligand_pairs(atom_pairs)}


def _locate_relax_input_model(rosetta_folder: str, model: str) -> Optional[Path]:
    """
    Locate the Rosetta input PDB corresponding to `model` inside `rosetta_folder`.
    """
    input_dir = Path(rosetta_folder) / "input_models"
    candidates = [
        input_dir / f"{model}.pdb",
        input_dir / f"{model}_INPUT.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_input_model_ligand_atoms(pdb_path: Path, ligand: str) -> Dict[str, Tuple[str, int, str]]:
    """
    Build a map of atom-name -> (chain, residue_number, atom_name) entries for `ligand`.
    """
    mapping: Dict[str, Tuple[str, int, str]] = {}
    if not pdb_path or not pdb_path.exists():
        return mapping

    ligand_name = ligand.strip().upper()
    if not ligand_name:
        return mapping

    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if resname != ligand_name:
                continue
            atom_name = line[12:16].strip()
            if not atom_name:
                continue
            chain = line[21].strip() or " "
            resseq_str = line[22:26].strip()
            try:
                resseq = int(resseq_str)
            except ValueError:
                continue
            mapping[atom_name] = (chain, resseq, atom_name)

    return mapping


def _normalize_relax_atom_pairs(atom_pairs: Dict[str, Any], rosetta_folder: str) -> Dict[str, List[List[Any]]]:
    """
    Normalise atom pair definitions for relax calculations.

    Supports either the legacy structure
        {model: [((chain, resid, atom), (chain, resid, atom)), ...]}
    or a ligand-aware structure
        {model: {ligand: [((chain, resid, atom), ligand_atom_name_or_tuple), ...]}}
    where ligand atoms can be specified by their atom names. In the latter case the
    atom coordinates are resolved from the Rosetta input PDB files.
    """

    if not isinstance(atom_pairs, dict):
        raise TypeError("atom_pairs must be a dictionary keyed by model name.")

    def _normalize_atom_spec(atom_spec):
        if (
            isinstance(atom_spec, (list, tuple))
            and len(atom_spec) == 3
        ):
            chain = str(atom_spec[0]).strip() or " "
            atom_name = str(atom_spec[2]).strip()
            if not atom_name:
                raise ValueError("Atom names in atom_pairs cannot be empty.")
            try:
                resseq = int(atom_spec[1])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Residue index '{atom_spec[1]}' in atom_pairs should be an integer."
                )
            return [chain, resseq, atom_name]
        raise ValueError(
            "Each atom_pairs entry must be defined as (chain, residue_index, atom_name)."
        )

    normalized: Dict[str, List[List[Any]]] = {}
    missing_atoms: Dict[Tuple[str, str], Set[str]] = {}
    pdb_cache: Dict[str, Optional[Path]] = {}
    ligand_atom_cache: Dict[Tuple[str, str], Dict[str, Tuple[str, int, str]]] = {}

    for model, entries in atom_pairs.items():
        if isinstance(entries, dict):
            expanded: List[List[Any]] = []
            for ligand, pair_list in entries.items():
                if not pair_list:
                    continue
                mapping = None

                for protein_atom, ligand_atom in pair_list:
                    protein_spec = _normalize_atom_spec(protein_atom)

                    if isinstance(ligand_atom, str):
                        if mapping is None:
                            cache_key = (model, ligand)
                            if cache_key not in ligand_atom_cache:
                                if model not in pdb_cache:
                                    pdb_cache[model] = _locate_relax_input_model(rosetta_folder, model)
                                pdb_path = pdb_cache[model]
                                if pdb_path is None:
                                    raise FileNotFoundError(
                                        f"Could not locate input model for '{model}' inside "
                                        f"{rosetta_folder}/input_models."
                                    )
                                ligand_atom_cache[cache_key] = _parse_input_model_ligand_atoms(
                                    pdb_path, ligand
                                )
                            mapping = ligand_atom_cache[cache_key]

                        if not mapping:
                            missing_atoms.setdefault((model, ligand), set()).add("<ligand_not_found>")
                            continue

                        ligand_tuple = mapping.get(ligand_atom)
                        if ligand_tuple is None:
                            missing_atoms.setdefault((model, ligand), set()).add(ligand_atom)
                            continue
                        ligand_spec = list(ligand_tuple)
                    else:
                        ligand_spec = _normalize_atom_spec(ligand_atom)

                    expanded.append([protein_spec, ligand_spec])

            if expanded or model not in normalized:
                normalized[model] = expanded

        else:
            pair_list = entries or []
            normalized_pairs: List[List[Any]] = []
            for pair in pair_list:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(
                        "atom_pairs entries must be 2-tuples: (protein_atom, ligand_atom)."
                    )
                normalized_pairs.append([
                    _normalize_atom_spec(pair[0]),
                    _normalize_atom_spec(pair[1]),
                ])
            normalized[model] = normalized_pairs

    if missing_atoms:
        missing_messages = []
        for (model, ligand), atoms in sorted(missing_atoms.items()):
            if "<ligand_not_found>" in atoms:
                missing_messages.append(
                    f"{model}: ligand '{ligand}' was not found in the Rosetta input model."
                )
                atoms = atoms - {"<ligand_not_found>"}
            if atoms:
                atom_list = ", ".join(sorted(atoms))
                missing_messages.append(f"{model}/{ligand}: missing atoms {atom_list}")
        raise ValueError(
            "Unable to match the requested ligand atoms against the Rosetta input models:\n"
            + "\n".join(missing_messages)
        )

    return normalized


def _prepare_expanded_atom_pairs(
    atom_pairs,
    docking_folder,
    model_ligands,
    separator,
    ligand_chain_overrides=None,
):
    """Return the full atom pair list expected by the PyRosetta script."""
    ligand_atom_cache: Dict[Tuple[str, Optional[str], Optional[int]], Dict[str, Tuple[str, int, str]]] = {}
    expanded = {}
    missing_ligand_atoms: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    models_without_pairs = []
    if ligand_chain_overrides is None:
        ligand_chain_overrides = {}

    def _resolve_ligand_tuple(ligand: str, atom_identifier, override):
        if isinstance(atom_identifier, tuple):
            return atom_identifier
        if not isinstance(atom_identifier, str):
            raise TypeError("Ligand atom identifier must be a string or tuple.")
        chain_override = override[0] if override else None
        resseq_override = override[1] if override else None
        cache_key = (ligand, chain_override, resseq_override)
        if cache_key not in ligand_atom_cache:
            ligand_atom_cache[cache_key] = _parse_ligand_atom_map(
                docking_folder, ligand, chain_override, resseq_override
            )
        mapping = ligand_atom_cache[cache_key]
        if atom_identifier not in mapping:
            available = _list_ligand_atom_names(docking_folder, ligand)
            suggestion = ", ".join(available[:10])
            raise KeyError(
                f"Atom '{atom_identifier}' not found in ligand {ligand}. "
                f"Available ligand atoms: {suggestion}{'...' if len(available)>10 else ''}"
            )
        return mapping[atom_identifier]

    for model_ligand in model_ligands:
        base_model, ligand = model_ligand.split(separator, 1)
        model_atom_pairs = atom_pairs.get(base_model)
        if not isinstance(model_atom_pairs, dict):
            models_without_pairs.append(model_ligand)
            continue
        raw_pairs = model_atom_pairs.get(ligand)
        if raw_pairs is None:
            models_without_pairs.append(model_ligand)
            continue

        override = ligand_chain_overrides.get(model_ligand)
        normalized_pairs = []
        for entry in raw_pairs:
            if len(entry) != 2:
                raise ValueError("Each atom pair entry must be a 2-tuple.")
            protein_atom, ligand_atom = entry
            if not (isinstance(protein_atom, tuple) and len(protein_atom) == 3):
                raise ValueError("Protein atom must be a tuple (chain, residue, atom).")
            try:
                ligand_tuple = _resolve_ligand_tuple(ligand, ligand_atom, override)
            except KeyError:
                if isinstance(ligand_atom, str):
                    missing_ligand_atoms[(base_model, ligand)].add(ligand_atom)
                continue
            normalized_pairs.append((protein_atom, ligand_tuple))

        if normalized_pairs:
            expanded[model_ligand] = normalized_pairs
        else:
            models_without_pairs.append(model_ligand)

    return expanded, missing_ligand_atoms, models_without_pairs

def _analyze_mapping_pairs(mapping, r_ca_coord, t_ca_coord, max_ca_ca, verbose_limit=40):
    import numpy as np, bisect
    if not mapping:
        print("[mapdiag] no pairs")
        return
    pairs = sorted([(r, t) for t, r in mapping.items()], key=lambda x: x[0])
    tgt_order = [t for _, t in pairs]

    inversions = 0
    last = -1
    for t in tgt_order:
        if t <= last:
            inversions += 1
        last = t

    lis = []
    for t in tgt_order:
        k = bisect.bisect_left(lis, t)
        if k == len(lis):
            lis.append(t)
        else:
            lis[k] = t
    lis_len = len(lis)

    d = []
    for t, r in mapping.items():
        d.append(float(np.linalg.norm(t_ca_coord[t] - r_ca_coord[r])))
    d = np.array(d, float)
    d_med = float(np.median(d)) if d.size else float("nan")
    d_max = float(np.max(d)) if d.size else float("nan")
    frac_loose = float((d > 0.75 * max_ca_ca).sum() / d.size) if d.size else float("nan")

    print(f"[mapdiag] pairs={len(mapping)}  inversions={inversions}  LIS_len={lis_len}")
    print(f"[mapdiag] dists: median={d_med:.2f}  max={d_max:.2f}  frac(>0.75*cutoff)={frac_loose:.2f}")

    for i, (r, t) in enumerate(pairs[:verbose_limit]):
        print(f"  {i:03d}: ref={r} -> tgt={t}  | d={np.linalg.norm(t_ca_coord[t]-r_ca_coord[r]):.2f}")

_MISSING = object()


class _OpenMMSimulationRegistry(dict):
    """Dictionary storing per-model openmm_md objects plus a command log registry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openmm_command_logs = {}


def _require_openmm_support(feature="OpenMM-dependent functionality"):
    """
    Import the OpenMM integration module on demand, raising a clear error when unavailable.
    """
    try:
        module = importlib.import_module("prepare_proteins.MD.openmm_setup")
    except Exception as exc:
        raise ImportError(
            f"OpenMM support is required to use {feature}. "
            "Install the 'openmm' package (e.g. `pip install openmm`) to enable this functionality."
        ) from exc
    if not getattr(module, "OPENMM_AVAILABLE", False):
        import_error = getattr(module, "OPENMM_IMPORT_ERROR", None)
        raise ImportError(
            f"OpenMM support is required to use {feature}. "
            "Install the 'openmm' package (e.g. `pip install openmm`) to enable this functionality."
        ) from import_error
    return module


import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from Bio import PDB, BiopythonWarning, pairwise2
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import aa3, is_aa
from ipywidgets import interactive_output, VBox, IntSlider, Checkbox, interact, fixed, Dropdown, FloatSlider, FloatRangeSlider
from pkg_resources import Requirement, resource_listdir, resource_stream
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

import prepare_proteins
from . import MD, _atom_selectors, alignment, rosettaScripts
from .analysis import find_neighbours_in_pdb
from .MD.parameterization import get_backend
from .MD.parameterization.utils import DEFAULT_PARAMETERIZATION_SKIP_RESIDUES


try:
    _BIOPYTHON_THREE_TO_ONE = PDB.Polypeptide.three_to_one
    def _three_to_one(resname):
        return _BIOPYTHON_THREE_TO_ONE(resname)

    _BIOPYTHON_ONE_TO_THREE = PDB.Polypeptide.one_to_three

    def _one_to_three(rescode):
        return _BIOPYTHON_ONE_TO_THREE(rescode).upper()

except AttributeError:
    from Bio.Data.PDBData import protein_letters_3to1_extended as _protein_letters_3to1_extended
    from Bio.Data.IUPACData import protein_letters_1to3 as _protein_letters_1to3

    def _three_to_one(resname):
        """Biopython>=1.83 compatibility: map three-letter codes to one letter."""
        key = f"{resname}".strip().upper()
        if not key:
            raise KeyError(resname)
        key = f"{key:<3s}"[:3]
        if key in _protein_letters_3to1_extended:
            return _protein_letters_3to1_extended[key]
        raise KeyError(resname)

    def _one_to_three(rescode):
        """Map one-letter amino acid code to three-letter code."""
        key = f"{rescode}".strip().upper()
        if len(key) != 1 or key not in _protein_letters_1to3:
            raise KeyError(rescode)
        return _protein_letters_1to3[key].upper()


class MutationVariabilityAnalyzer:
    """Summarise mutational variability across models relative to a wild-type reference."""

    _SPECIAL_RESNAME_MAP = {
        "HIE": "HIS",
        "HID": "HIS",
        "HIP": "HIS",
        "CYX": "CYS",
        "ASH": "ASP",
        "GLH": "GLU",
    }

    _DEFAULT_ALIGNMENT = {
        "match": 2.0,
        "mismatch": -1.0,
        "gap_open": -10.0,
        "gap_extend": -0.5,
    }

    _MUTATION_COLUMNS = [
        "Model",
        "Chain",
        "Category",
        "ChangeKind",
        "Mutation",
        "WT_residue",
        "Mutant_residue",
        "WT_resseq",
        "WT_icode",
        "Variant_resseq",
        "Variant_icode",
        "WT_position",
        "Variant_position",
        "Location",
        "Contact_total",
        "Ligand_contacts",
        "SiteID",
    ]

    _HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "W", "Y", "P"}
    _POLAR = {"S", "T", "N", "Q", "C"}
    _CHARGED = {"K", "R", "H", "D", "E"}
    _SPECIAL = {"G"}

    _CATEGORY_COLOR_MAP = {
        "HH": "#1f77b4",
        "HP": "#ff7f0e",
        "PH": "#2ca02c",
        "PP": "#d62728",
        "HC": "#9467bd",
        "CH": "#8c564b",
        "CC": "#e377c2",
        "PC": "#7f7f7f",
        "CP": "#bcbd22",
        "HS": "#17becf",
        "SH": "#393b79",
        "PS": "#ff9896",
        "SP": "#98df8a",
        "CS": "#c5b0d5",
        "SC": "#c49c94",
        "SS": "#f7b6d2",
        "HU": "#9edae5",
        "UH": "#ad494a",
        "PU": "#a55194",
        "UP": "#637939",
        "CU": "#8ca252",
        "UC": "#bd9e39",
    }

    def __init__(
        self,
        protein_models: "proteinModels",
        wild_type: str,
        *,
        only_models: Optional[Union[str, Iterable[str]]] = None,
        exclude_models: Optional[Union[str, Iterable[str]]] = None,
        chains: Optional[Union[str, Iterable[str], Dict[str, Iterable[str]]]] = None,
        alignment_params: Optional[Dict[str, float]] = None,
        residue_annotations: Optional[Union[Dict[Any, str], pd.DataFrame]] = None,
        contact_results: Optional[Dict[str, Any]] = None,
        auto_classify_locations: bool = True,
        core_contact_threshold: int = 12,
        surface_contact_threshold: int = 4,
        pocket_neighbor_kinds: Optional[Iterable[str]] = None,
    ):
        self.models = protein_models
        available_models = list(getattr(self.models, "models_names", []))
        if wild_type not in available_models:
            raise ValueError(f"Wild-type model '{wild_type}' is not available in the current session.")
        self.wt_model = wild_type

        # Determine the subset of models included in the analysis
        def _normalize_model_list(value, label):
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple, set)):
                return [str(v) for v in value]
            raise TypeError(f"{label} must be a string or an iterable of strings.")

        only_list = _normalize_model_list(only_models, "only_models")
        exclude_list = _normalize_model_list(exclude_models, "exclude_models")

        selected = list(available_models)
        if only_list:
            only_set = set(only_list)
            selected = [m for m in selected if m in only_set]
        if exclude_list:
            exclude_set = set(exclude_list)
            selected = [m for m in selected if m not in exclude_set]

        if self.wt_model not in selected:
            raise ValueError(
                "Wild-type model must be part of the analysis set; "
                "adjust only_models/exclude_models so that it is included."
            )

        if not selected:
            raise ValueError("No models left after applying only_models/exclude_models filters.")

        self.analysis_models: List[str] = selected

        self._align_params = dict(self._DEFAULT_ALIGNMENT)
        if alignment_params:
            self._align_params.update({k: float(v) for k, v in alignment_params.items() if k in self._DEFAULT_ALIGNMENT})

        if self.models.chain_sequences == {} and getattr(self.models, "structures", None):
            self.models.getModelsSequences()

        raw_selection = self.models._resolve_chain_selection(chains=chains, models=self.analysis_models)
        wt_chains = raw_selection.get(self.wt_model, [])
        if not wt_chains:
            raise ValueError(f"No analyzable chains were found for the wild-type model '{self.wt_model}'.")
        wt_set = set(wt_chains)

        self.chain_selection: Dict[str, List[str]] = {}
        dropped_models: List[str] = []
        for model, model_chains in raw_selection.items():
            shared = [c for c in model_chains if c in wt_set]
            if not shared:
                dropped_models.append(model)
                continue
            self.chain_selection[model] = shared
        if dropped_models:
            warnings.warn(
                "Ignoring models without overlapping chains relative to the WT selection: "
                + ", ".join(sorted(dropped_models)),
                RuntimeWarning,
            )

        self.core_contact_threshold = max(int(core_contact_threshold), 0)
        self.surface_contact_threshold = max(int(surface_contact_threshold), 0)
        if self.core_contact_threshold < self.surface_contact_threshold:
            self.core_contact_threshold = self.surface_contact_threshold

        default_pocket = {"organic", "cofactor", "ligand", "metal", "ion"}
        if pocket_neighbor_kinds is None:
            self.pocket_neighbor_kinds = default_pocket
        else:
            self.pocket_neighbor_kinds = {str(kind).lower() for kind in pocket_neighbor_kinds}
            if not self.pocket_neighbor_kinds:
                self.pocket_neighbor_kinds = default_pocket

        self._chain_sequences: Dict[str, Dict[str, str]] = {}
        self._chain_metadata: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._model_residue_counts: Dict[str, int] = {}

        for model, chains_for_model in self.chain_selection.items():
            per_chain_sequences = {}
            per_chain_metadata = {}
            for chain_id in chains_for_model:
                seq, metadata = self._extract_chain_sequence(model, chain_id)
                if not seq:
                    continue
                per_chain_sequences[chain_id] = seq
                per_chain_metadata[chain_id] = metadata
            if per_chain_sequences:
                self._chain_sequences[model] = per_chain_sequences
                self._chain_metadata[model] = per_chain_metadata
                self._model_residue_counts[model] = sum(len(meta) for meta in per_chain_metadata.values())

        if self.wt_model not in self._chain_sequences:
            raise ValueError("Wild-type chains could not be parsed; ensure the reference contains protein residues.")
        self._wt_chain_sequences = self._chain_sequences[self.wt_model]

        contact_results_all = contact_results or getattr(self.models, "models_contacts", None) or {}
        if contact_results_all:
            filtered_contacts = {
                model: result
                for model, result in contact_results_all.items()
                if model in self.chain_selection
            }
        else:
            filtered_contacts = {}
        self._contact_summary = self._build_contact_summary(filtered_contacts)

        self._location_map = self._normalize_location_annotations(residue_annotations)
        if auto_classify_locations and self._contact_summary:
            self.infer_locations_from_contacts(overwrite=False)

        self._mutation_table: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_mutation_table(
        self,
        models: Optional[Union[str, Iterable[str]]] = None,
        *,
        include_wt: bool = False,
        recompute: bool = False,
    ) -> pd.DataFrame:
        """Return a per-mutation table for the selected models."""

        if self._mutation_table is None or recompute:
            self._mutation_table = self._compute_mutation_table()

        df = self._mutation_table
        selected = set(self._select_models(models=models, include_wt=include_wt))
        if not selected:
            return df.iloc[0:0].copy()

        mask = df["Model"].isin(selected)
        return df.loc[mask].reset_index(drop=True)

    def summarize_models(
        self,
        models: Optional[Union[str, Iterable[str]]] = None,
        *,
        include_wt: bool = False,
    ) -> pd.DataFrame:
        """Return per-model mutation counts, percentages, and load metrics."""

        selected_models = self._select_models(models=models, include_wt=include_wt)
        table = self.build_mutation_table(models=selected_models, include_wt=include_wt)

        if table.empty:
            summary = pd.DataFrame({"Model": selected_models})
            summary["total_mutations"] = 0
            summary["substitutions"] = 0
            summary["insertions"] = 0
            summary["deletions"] = 0
            summary["unique_sites"] = 0
        else:
            classified = table.assign(
                is_substitution=table["ChangeKind"].eq("substitution"),
                is_insertion=table["ChangeKind"].eq("insertion"),
                is_deletion=table["ChangeKind"].eq("deletion"),
            )
            summary = (
                classified.groupby("Model")
                .agg(
                    total_mutations=("Mutation", "count"),
                    substitutions=("is_substitution", "sum"),
                    insertions=("is_insertion", "sum"),
                    deletions=("is_deletion", "sum"),
                )
                .reset_index()
            )
            unique_sites = (
                table.loc[table["SiteID"].notna()]
                .groupby("Model")["SiteID"]
                .nunique()
                .reset_index(name="unique_sites")
            )
            summary = summary.merge(unique_sites, on="Model", how="left").fillna({"unique_sites": 0})
            summary["unique_sites"] = summary["unique_sites"].astype(int)

        summary = summary.set_index("Model").reindex(selected_models).fillna(0).reset_index()
        summary["total_residues"] = summary["Model"].map(self._model_residue_counts).fillna(0).astype(int)
        summary["percent_mutated"] = summary.apply(
            lambda row: (row["unique_sites"] / row["total_residues"]) * 100 if row["total_residues"] else 0.0,
            axis=1,
        )
        columns = [
            "Model",
            "total_mutations",
            "percent_mutated",
            "unique_sites",
            "substitutions",
            "insertions",
            "deletions",
            "total_residues",
        ]
        return summary[columns]

    def mutation_type_counts(
        self,
        models: Optional[Union[str, Iterable[str]]] = None,
        *,
        collapse_positions: bool = False,
    ) -> pd.DataFrame:
        """Return counts of mutation types per model."""

        table = self.build_mutation_table(models=models)
        subs = table[table["ChangeKind"].eq("substitution")].copy()
        if subs.empty:
            columns = ["Model", "Mutation", "Count"] if not collapse_positions else ["Model", "From", "To", "Count"]
            return pd.DataFrame(columns=columns)

        if collapse_positions:
            subs["From"] = subs["WT_residue"].fillna("?")
            subs["To"] = subs["Mutant_residue"].fillna("?")
            counts = (
                subs.groupby(["Model", "From", "To"], dropna=False)
                .size()
                .reset_index(name="Count")
                .sort_values(["Model", "Count"], ascending=[True, False])
            )
            return counts

        counts = (
            subs.groupby(["Model", "Mutation"], dropna=False)
            .size()
            .reset_index(name="Count")
            .sort_values(["Model", "Count"], ascending=[True, False])
        )
        return counts

    def mutation_presence_matrix(
        self,
        models: Optional[Union[str, Iterable[str]]] = None,
        *,
        include_insertions: bool = False,
        include_deletions: bool = False,
        model_order: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Return a mutation (rows) × model (columns) presence/absence matrix."""

        table = self.build_mutation_table(models=models)
        if table.empty:
            return pd.DataFrame()

        allowed = {"substitution"}
        if include_insertions:
            allowed.add("insertion")
        if include_deletions:
            allowed.add("deletion")

        data = table.reset_index(drop=False)
        data = data[data["ChangeKind"].isin(allowed)].copy()
        if data.empty:
            return pd.DataFrame(columns=sorted(data["Model"].unique()))

        data["MutationLabel"] = data.apply(self._format_mutation_label, axis=1)

        matrix = (
            data.assign(value=1)
            .groupby(["MutationLabel", "Model"])["value"]
            .max()
            .unstack(fill_value=0)
            .astype(int)
        )

        if model_order is not None:
            ordered = [m for m in model_order if m in matrix.columns]
            remaining = [c for c in matrix.columns if c not in ordered]
            matrix = matrix[ordered + remaining]

        totals = matrix.sum(axis=1)
        order = totals.sort_values(ascending=False).index
        matrix = matrix.loc[order]
        matrix["#Designs"] = totals.loc[order].astype(int)

        design_totals = matrix.drop(columns=["#Designs"], errors="ignore").sum(axis=0).astype(int)
        summary_row = design_totals.rename("#Mutations")
        summary_row["#Designs"] = int(matrix["#Designs"].sum())
        matrix = pd.concat([matrix, summary_row.to_frame().T])
        return matrix

    def plot_mutation_presence_matrix(
        self,
        models: Optional[Union[str, Iterable[str]]] = None,
        *,
        include_insertions: bool = False,
        include_deletions: bool = False,
        ax: Optional[plt.Axes] = None,
        cmap: str = "Blues",
        model_order: Optional[Iterable[str]] = None,
        color_by_category: bool = False,
        dpi: Optional[float] = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Visualise the mutation presence matrix as a heatmap."""

        matrix = self.mutation_presence_matrix(
            models=models,
            include_insertions=include_insertions,
            include_deletions=include_deletions,
        )
        if matrix.empty:
            if ax is None:
                _, ax = plt.subplots(figsize=(4, 2), dpi=dpi)
            ax.set_axis_off()
            ax.set_title("No mutations to display")
            return ax

        heatmap_data = matrix.drop(index=["#Mutations"], errors="ignore").drop(columns=["#Designs"], errors="ignore")
        if model_order is not None:
            order = [m for m in model_order if m in heatmap_data.columns]
            remaining = [c for c in heatmap_data.columns if c not in order]
            heatmap_data = heatmap_data[order + remaining]
        if heatmap_data.empty:
            if ax is None:
                _, ax = plt.subplots(figsize=(4, 2), dpi=dpi)
            ax.set_axis_off()
            ax.set_title("No mutations to display")
            return ax

        if ax is None:
            height = max(3, heatmap_data.shape[0] * 0.25)
            width = max(3, heatmap_data.shape[1] * 0.4)
            _, ax = plt.subplots(figsize=(width, height), dpi=dpi)

        if color_by_category:
            table = self.build_mutation_table(models=models)
            allowed = {"substitution"}
            if include_insertions:
                allowed.add("insertion")
            if include_deletions:
                allowed.add("deletion")
            table_df = table.reset_index().copy()
            table_df = table_df[table_df["ChangeKind"].isin(allowed)]
            table_df["MutationLabel"] = table_df.apply(self._format_mutation_label, axis=1)
            category_lookup = {
                (row.MutationLabel, row.Model): row.Category
                for row in table_df.itertuples()
            }
            categories = sorted({cat for cat in category_lookup.values() if cat})
            if not categories:
                color_by_category = False
            else:
                cat_to_idx = {cat: idx + 1 for idx, cat in enumerate(categories)}
                color_matrix = np.zeros(heatmap_data.shape, dtype=int)
                for i, mut in enumerate(heatmap_data.index):
                    for j, model_name in enumerate(heatmap_data.columns):
                        if heatmap_data.iat[i, j]:
                            cat = category_lookup.get((mut, model_name))
                            if cat:
                                color_matrix[i, j] = cat_to_idx.get(cat, 0)
                colors = ["#f8f8f8"]
                palette = [
                    "#393b79",
                    "#637939",
                    "#8c6d31",
                    "#843c39",
                    "#7b4173",
                    "#3182bd",
                    "#e6550d",
                    "#31a354",
                    "#756bb1",
                    "#636363",
                ]
                if isinstance(cmap, str):
                    base = mpl.colormaps.get_cmap(cmap) if hasattr(mpl, "colormaps") else plt.get_cmap(cmap)
                else:
                    base = cmap

                assigned: List[str] = []
                palette_idx = 0
                for cat in categories:
                    color = self._CATEGORY_COLOR_MAP.get(cat)
                    if color is None:
                        if palette_idx < len(palette):
                            color = palette[palette_idx]
                            palette_idx += 1
                        else:
                            sample_pos = 0.1 + 0.8 * (palette_idx - len(palette)) / max(len(categories), 1)
                            color = base(sample_pos)
                            palette_idx += 1
                    assigned.append(color)
                colors.extend(assigned)
                categorical_cmap = mpl.colors.ListedColormap(colors)
                bounds = np.arange(len(colors) + 1) - 0.5
                im = ax.imshow(color_matrix, aspect="auto", cmap=categorical_cmap, vmin=-0.5, vmax=len(colors) - 1.5)
                cbar = plt.colorbar(
                    im,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04,
                    boundaries=bounds,
                    ticks=np.arange(len(colors)),
                )
                cbar.set_ticklabels(["absent"] + categories)
                cbar.set_label("Mutation class")

        if not color_by_category:
            if isinstance(cmap, str):
                base_cmap = mpl.colormaps.get_cmap(cmap) if hasattr(mpl, "colormaps") else plt.get_cmap(cmap)
            else:
                base_cmap = cmap
            discrete_cmap = mpl.colors.ListedColormap([base_cmap(0.0), base_cmap(0.8)])
            im = ax.imshow(heatmap_data.values, aspect="auto", cmap=discrete_cmap, vmin=0, vmax=1)
            cbar = plt.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                boundaries=[-0.5, 0.5, 1.5],
                ticks=[0, 1],
            )
            cbar.set_ticklabels(["absence", "presence"])

        ax.set_xticks(np.arange(heatmap_data.shape[1]))
        ax.set_xticklabels(heatmap_data.columns, rotation=90, ha="center")
        ax.set_yticks(np.arange(heatmap_data.shape[0]))
        ax.set_yticklabels(heatmap_data.index)
        ax.set_xlabel("Model")
        ax.set_ylabel("Mutation")
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        return ax

    def location_breakdown(
        self,
        models: Optional[Union[str, Iterable[str]]] = None,
        *,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Return counts (or fractions) of mutations per location class."""

        table = self.build_mutation_table(models=models)
        if table.empty:
            cols = ["Model", "Location", "Count"]
            if normalize:
                cols.append("Fraction")
            return pd.DataFrame(columns=cols)

        grouped = (
            table.groupby(["Model", "Location"], dropna=False)
            .size()
            .reset_index(name="Count")
        )
        if normalize:
            totals = grouped.groupby("Model")["Count"].transform(lambda x: x.sum() if x.sum() else 1)
            grouped["Fraction"] = grouped["Count"] / totals
        return grouped

    def plot_mutation_counts(
        self,
        models: Optional[Union[str, Iterable[str]]] = None,
        *,
        include_percent: bool = False,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Bar plot showing the number of mutations per model."""

        summary = self.summarize_models(models=models)
        if ax is None:
            _, ax = plt.subplots(figsize=(max(4, len(summary) * 0.6), 4))

        x = np.arange(len(summary))
        ax.bar(x, summary["total_mutations"], color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["Model"], rotation=45, ha="right")
        ax.set_ylabel("Number of mutations")
        ax.set_title("Mutation load per model")

        if include_percent:
            for xpos, (_, row) in zip(x, summary.iterrows()):
                ax.text(
                    xpos,
                    row["total_mutations"] + 0.05,
                    f"{row['percent_mutated']:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.margins(y=0.2)
        return ax

    def plot_location_breakdown(
        self,
        models: Optional[Union[str, Iterable[str]]] = None,
        *,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Stacked bar chart showing mutation locations (core/surface/pocket)."""

        breakdown = self.location_breakdown(models=models)
        if ax is None:
            _, ax = plt.subplots(figsize=(max(4, len(breakdown["Model"].unique()) * 0.6), 4))

        if breakdown.empty:
            ax.set_axis_off()
            ax.set_title("No mutations to plot")
            return ax

        pivot = breakdown.pivot(index="Model", columns="Location", values="Count").fillna(0)
        models_idx = pivot.index.tolist()
        x = np.arange(len(models_idx))
        bottom = np.zeros(len(models_idx))
        for location in pivot.columns:
            values = pivot[location].values
            ax.bar(x, values, bottom=bottom, label=location)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(models_idx, rotation=45, ha="right")
        ax.set_ylabel("Mutation count")
        ax.set_title("Mutation locations")
        ax.legend()
        ax.margins(y=0.1)
        return ax

    def update_residue_annotations(
        self,
        annotations: Union[Dict[Any, str], pd.DataFrame],
        *,
        overwrite: bool = True,
    ) -> None:
        """Update manual residue location annotations."""

        normalized = self._normalize_location_annotations(annotations)
        for key, value in normalized.items():
            if not overwrite and key in self._location_map:
                continue
            self._location_map[key] = value

    def infer_locations_from_contacts(self, *, overwrite: bool = False) -> None:
        """Populate location map based on contact heuristics."""

        for model, residues in self._contact_summary.items():
            for (chain, resseq, icode), stats in residues.items():
                location = self._classify_location_from_stats(stats)
                key = (model, chain, resseq, icode)
                if not location:
                    continue
                if not overwrite and key in self._location_map:
                    continue
                self._location_map[key] = location

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_models(
        self,
        models: Optional[Union[str, Iterable[str]]],
        *,
        include_wt: bool = False,
    ) -> List[str]:
        available = [m for m in self._chain_sequences.keys() if include_wt or m != self.wt_model]
        if models is None:
            return available
        if isinstance(models, str):
            requested = [models]
        else:
            requested = [str(m) for m in models]
        missing = [m for m in requested if m not in available and not (include_wt and m == self.wt_model)]
        if missing:
            warnings.warn(
                "Requested models were not part of the analysis set: " + ", ".join(sorted(missing)),
                RuntimeWarning,
            )
        return [m for m in requested if (m in available) or (include_wt and m == self.wt_model)]

    def _extract_chain_sequence(self, model: str, chain_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        sequence: List[str] = []
        metadata: List[Dict[str, Any]] = []
        structure = self.models.structures.get(model)
        if structure is None:
            return "", []
        chain_obj = None
        for pdb_model in structure:
            if chain_id in pdb_model.child_dict:
                chain_obj = pdb_model[chain_id]
                break
        if chain_obj is None:
            warnings.warn(f"Chain {chain_id} not found in model {model}; skipping.", RuntimeWarning)
            return "", []
        seq_idx = 0
        for residue in chain_obj:
            if residue.id[0] != " ":
                continue
            resname = self._SPECIAL_RESNAME_MAP.get(residue.resname.strip().upper(), residue.resname.strip().upper())
            try:
                one_letter = _three_to_one(resname)
            except Exception:
                one_letter = "X"
            seq_idx += 1
            metadata.append(
                {
                    "seq_index": seq_idx,
                    "resseq": int(residue.id[1]),
                    "icode": (residue.id[2] or " ").strip() or " ",
                    "resname": resname,
                    "chain": chain_id,
                }
            )
            sequence.append(one_letter)
        return "".join(sequence), metadata

    def _build_contact_summary(self, contact_results: Dict[str, Any]) -> Dict[str, Dict[Tuple[str, int, str], Dict[str, Any]]]:
        summary: Dict[str, Dict[Tuple[str, int, str], Dict[str, Any]]] = {}
        if not contact_results:
            return summary

        for model, result in contact_results.items():
            table = None
            if isinstance(result, pd.DataFrame):
                table = result
            elif isinstance(result, dict):
                table = result.get("table") if isinstance(result.get("table"), pd.DataFrame) else None
            if table is None or table.empty:
                continue
            df = table.copy()
            if "query_chain" not in df.columns or "query_resseq" not in df.columns:
                continue
            if "n_atom_contacts" not in df.columns:
                df["n_atom_contacts"] = 1
            per_residue: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
            for row in df.itertuples():
                chain = str(row.query_chain)
                resseq = int(row.query_resseq)
                icode = getattr(row, "query_icode", " ") or " "
                key = (chain, resseq, icode)
                entry = per_residue.setdefault(
                    key,
                    {
                        "total_contacts": 0,
                        "ligand_contacts": 0,
                        "neighbor_kinds": set(),
                        "min_distance": float("inf"),
                    },
                )
                contacts = getattr(row, "n_atom_contacts", 1) or 1
                entry["total_contacts"] += contacts
                neighbor_kind = str(getattr(row, "neighbor_kind", "")).lower()
                if neighbor_kind in self.pocket_neighbor_kinds:
                    entry["ligand_contacts"] += contacts
                entry["neighbor_kinds"].add(neighbor_kind)
                entry["min_distance"] = min(entry["min_distance"], float(getattr(row, "min_distance", np.inf)))
            if per_residue:
                summary[model] = per_residue
        return summary

    def _normalize_location_annotations(
        self, annotations: Optional[Union[Dict[Any, str], pd.DataFrame]]
    ) -> Dict[Tuple[str, str, int, str], str]:
        mapping: Dict[Tuple[str, str, int, str], str] = {}
        if annotations is None:
            return mapping

        if isinstance(annotations, pd.DataFrame):
            required = {"Model", "Chain", "Resseq", "Location"}
            missing = required - set(annotations.columns)
            if missing:
                raise ValueError(f"Residue annotation DataFrame is missing columns: {', '.join(sorted(missing))}")
            for row in annotations.itertuples():
                key = (
                    str(row.Model),
                    str(row.Chain),
                    int(row.Resseq),
                    getattr(row, "Icode", " ") or " ",
                )
                mapping[key] = str(row.Location).strip().lower() or "unknown"
            return mapping

        if isinstance(annotations, dict):
            for raw_key, value in annotations.items():
                if not isinstance(raw_key, (list, tuple)) or len(raw_key) not in (3, 4):
                    raise ValueError("Residue annotation keys must be (model, chain, resseq[, icode]).")
                model, chain, resseq = raw_key[:3]
                icode = raw_key[3] if len(raw_key) == 4 else " "
                mapping[(str(model), str(chain), int(resseq), (icode or " "))] = str(value).strip().lower() or "unknown"
            return mapping

        raise TypeError("Residue annotations must be provided as a dict or a pandas DataFrame.")

    def _compute_mutation_table(self) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        models = self._select_models(models=None, include_wt=False)
        for model in models:
            for chain_id in self.chain_selection.get(model, []):
                if chain_id not in self._wt_chain_sequences:
                    continue
                wt_seq = self._wt_chain_sequences[chain_id]
                variant_seq = self._chain_sequences.get(model, {}).get(chain_id)
                if not variant_seq:
                    continue
                wt_meta = self._chain_metadata[self.wt_model][chain_id]
                var_meta = self._chain_metadata[model][chain_id]
                wt_aln, var_aln = self._align_sequences(wt_seq, variant_seq)
                records.extend(
                    self._records_from_alignment(model, chain_id, wt_aln, var_aln, wt_meta, var_meta)
                )
        if not records:
            empty_df = pd.DataFrame(columns=self._MUTATION_COLUMNS)
            return empty_df.set_index(pd.MultiIndex.from_arrays([[], []], names=["Model", "Chain"]))

        df = pd.DataFrame.from_records(records, columns=self._MUTATION_COLUMNS)
        return df.set_index(["Model", "Chain"], drop=False)

    def _align_sequences(self, wt_seq: str, variant_seq: str) -> Tuple[str, str]:
        alignment = pairwise2.align.globalms(
            wt_seq,
            variant_seq,
            self._align_params["match"],
            self._align_params["mismatch"],
            self._align_params["gap_open"],
            self._align_params["gap_extend"],
            one_alignment_only=True,
        )
        if not alignment:
            raise ValueError("Failed to align sequences; please verify the selected chains.")
        wt_aln, var_aln = alignment[0][:2]
        return wt_aln, var_aln

    def _records_from_alignment(
        self,
        model: str,
        chain_id: str,
        wt_aln: str,
        var_aln: str,
        wt_meta: List[Dict[str, Any]],
        var_meta: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        wt_idx = -1
        var_idx = -1
        for wt_res, var_res in zip(wt_aln, var_aln):
            wt_info = None
            var_info = None
            if wt_res != "-":
                wt_idx += 1
                wt_info = wt_meta[wt_idx] if wt_idx < len(wt_meta) else None
            if var_res != "-":
                var_idx += 1
                var_info = var_meta[var_idx] if var_idx < len(var_meta) else None
            if wt_res == var_res:
                continue
            change_kind = "substitution"
            if wt_res == "-":
                change_kind = "insertion"
            elif var_res == "-":
                change_kind = "deletion"

            mutation_str = self._format_mutation_string(wt_info, wt_res, var_res)
            location, stats = self._resolve_location_context(model, var_info, wt_info)
            category = self._mutation_category_label(wt_res, var_res)

            record = {
                "Model": model,
                "Chain": chain_id,
                "Category": category,
                "ChangeKind": change_kind,
                "Mutation": mutation_str,
                "WT_residue": wt_res,
                "Mutant_residue": var_res,
                "WT_resseq": wt_info["resseq"] if wt_info else None,
                "WT_icode": wt_info["icode"] if wt_info else None,
                "Variant_resseq": var_info["resseq"] if var_info else None,
                "Variant_icode": var_info["icode"] if var_info else None,
                "WT_position": wt_info["seq_index"] if wt_info else None,
                "Variant_position": var_info["seq_index"] if var_info else None,
                "Location": location or "unknown",
                "Contact_total": stats.get("total_contacts") if stats else None,
                "Ligand_contacts": stats.get("ligand_contacts") if stats else None,
                "SiteID": self._site_identifier(chain_id, wt_info),
            }
            records.append(record)
        return records

    def _format_mutation_string(
        self,
        wt_info: Optional[Dict[str, Any]],
        wt_res: str,
        var_res: str,
    ) -> str:
        pos = wt_info["resseq"] if wt_info else "?"
        icode = wt_info["icode"].strip() if wt_info and wt_info["icode"] else ""
        return f"{wt_res}{pos}{icode}{var_res}"

    def _format_mutation_label(self, row: pd.Series) -> str:  # type: ignore[name-defined]
        chain = row.get("Chain")
        mutation = row.get("Mutation")
        if mutation and len(mutation) >= 2:
            try:
                prefix = mutation[0]
                suffix = mutation[-1]
                digits = ''.join(ch for ch in mutation[1:-1] if ch.isdigit())
                mutation_simple = f"{prefix}{digits}{suffix}" if digits else mutation
            except Exception:
                mutation_simple = mutation
        else:
            mutation_simple = mutation or "?"
        return f"{chain}:{mutation_simple}"

    def _site_identifier(self, chain_id: str, meta: Optional[Dict[str, Any]]) -> Optional[str]:
        if not meta:
            return None
        return f"{chain_id}:{meta['resseq']}{meta['icode'].strip()}"

    def _resolve_location_context(
        self,
        model: str,
        variant_meta: Optional[Dict[str, Any]],
        wt_meta: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        for candidate_model, meta in (
            (model, variant_meta),
            (self.wt_model if model != self.wt_model else model, wt_meta),
        ):
            if meta is None:
                continue
            location, stats = self._location_and_stats(candidate_model, meta)
            if location or stats:
                return location, stats
        return None, None

    def _location_and_stats(
        self,
        model: str,
        meta: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        key = (model, meta["chain"], meta["resseq"], meta["icode"])
        location = self._location_map.get(key)
        stats = self._contact_summary.get(model, {}).get((meta["chain"], meta["resseq"], meta["icode"]))
        if location:
            return location, stats
        if stats:
            return self._classify_location_from_stats(stats), stats
        return None, None

    def _classify_location_from_stats(self, stats: Dict[str, Any]) -> Optional[str]:
        if stats.get("ligand_contacts", 0) > 0:
            return "pocket"
        total = stats.get("total_contacts", 0)
        if total >= self.core_contact_threshold:
            return "core"
        if total <= self.surface_contact_threshold:
            return "surface"
        return "boundary"

    def _residue_category_code(self, residue: str) -> str:
        if not residue or residue == "-":
            return "-"
        r = residue.upper()
        if r in self._HYDROPHOBIC:
            return "H"
        if r in self._POLAR:
            return "P"
        if r in self._CHARGED:
            return "C"
        if r in self._SPECIAL:
            return "S"
        return "U"

    def _mutation_category_label(self, wt_res: str, var_res: str) -> str:
        wt_code = self._residue_category_code(wt_res)
        var_code = self._residue_category_code(var_res)
        if "-" in (wt_code, var_code):
            return "gap"
        return f"{wt_code}{var_code}"

class proteinModels:
    """
    Attributes
    ==========
    models_folder : str
        Path to folder were PDB models are located.
    models_paths : dict
        Contains the paths to each model folder.
    msa :
        Multiple sequence alignment object
    multi_chain :
        Whether any model contains multiple chains
    sequences : dict
        Contains the sequence of each model.
    structures : dict
        Contains the Bio.PDB.Structure object to each model.

    Methods
    =======
    readModelFromPDB(self, models_folder)
        Read a model from a PDB file.
    getModelsSequences(self)
        Updates the sequences of each model in the PDB.
    calculateMSA(self)
        Calculates a multiple sequence alignment from the current sequences.
    calculateSecondaryStructure(self)
        Calculate secondary structure strings using DSSP.
    removeTerminalUnstructuredRegions(self)
        Remove terminal unstructured regions from all models.
    saveModels(self, output_folder)
        Save current models into an output folder.
    computeModelContacts(self, query_chains, **kwargs)
        Analyse inter-chain/ligand contacts for the selected models and cache the result.

    Hidden Methods
    ==============
    _getChainSequence(self, chain):
        Get the sequnce from a Bio.PDB.Chain.Chain object.
    _getModelsPaths(self)
        Get the paths for all PDBs in the input_folder path.
    """

    def __init__(
        self,
        models_folder,
        get_sequences=True,
        get_ss=False,
        msa=False,
        verbose=False,
        only_models=None,
        exclude_models=None,
        ignore_biopython_warnings=False,
        collect_memory_every=None,
        only_hetatm_conects=False,
        wat_to_hoh=True,
        maestro_export_options=None,
    ):
        """
        Read PDB models as Bio.PDB structure objects.

        Parameters
        ==========
        models_folder : str
            Path to the folder containing the PDB models.
        get_sequences : bool
            Get the sequences from the structure. They will be separated by chain and
            can be accessed through the .sequences attribute.
        get_ss : bool
            Get the strign representing the secondary structure of the models. they
            can be accessed through the .ss attribute.
        msa : bool
            single-chain structures at startup, othewise look for the calculateMSA()
            method.
        maestro_export_options : dict, optional
            Extra options passed to the Maestro-to-PDB export script when the input
            path is a .mae/.maegz file. Supported keys mirror the CLI arguments of
            `scripts/export_maestro_models.py` (e.g. prefix, protein_ct,
            ligand_chain, ligand_resnum, keep_original_ligand_ids, separator).
        """

        if ignore_biopython_warnings:
            warnings.simplefilter("ignore", BiopythonWarning)

        if only_models == None:
            only_models = []

        elif isinstance(only_models, str):
            only_models = [only_models]

        elif not isinstance(only_models, (list, tuple, set)):
            raise ValueError(
                "You must give models as a list or a single model as a string!"
            )

        if exclude_models == None:
            exclude_models = []

        elif isinstance(exclude_models, str):
            exclude_models = [exclude_models]

        elif not isinstance(exclude_models, (list, tuple, set)):
            raise ValueError(
                "You must give excluded models as a list or a single model as a string!"
            )

        self._maestro_temp_dir = None
        self._maestro_manifest = None
        self._maestro_manifest_path = None
        self._maestro_source_file = None
        self.maestro_export_options = maestro_export_options

        if isinstance(models_folder, str):
            models_folder = self._maybe_prepare_maestro_models(
                models_folder, maestro_export_options
            )

        self.models_folder = models_folder
        if isinstance(self.models_folder, dict):
            self.models_paths = self.models_folder
        elif isinstance(self.models_folder, str):
            self.models_paths = self._getModelsPaths(
                only_models=only_models, exclude_models=exclude_models
            )
        self.models_names = []  # Store model names
        self.structures = {}  # structures are stored here
        self.sequences = {}  # sequences are stored here (single-chain legacy access)
        self.chain_sequences = {}  # per-chain sequences for each model
        self.target_sequences = {}  # Final sequences are stored here
        self.msa = None  # multiple sequence alignment
        self.msa_chain_selection = {}  # chain mapping used for latest MSA
        self.multi_chain = False
        self.ss = {}  # secondary structure strings are stored here (legacy access)
        self.chain_secondary_structure = {}  # per-chain secondary structure strings
        self.docking_data = None  # secondary structure strings are stored here
        self.docking_distances = {}
        self.docking_angles = {}
        self.docking_metric_type = {}
        self.docking_ligands = {}
        self.rosetta_docking_data = None  # secondary structure strings are stored here
        self.rosetta_docking_distances = {}
        self.rosetta_docking_angles = {}
        self.rosetta_docking_metric_type = {}
        self.rosetta_docking_ligands = {}
        self.rosetta_data = None  # Rosetta data is stored here
        self.sequence_differences = {}  # Store missing/changed sequence information
        self.conects = {}  # Store the conection inforamtion for each model
        self.covalent = {}  # Store covalent residues

        self.distance_data = {}
        self.models_data = {}
        self.models_contacts = {}

        # Read PDB structures into Biopython (serial loading)
        collect_memory = False
        for i, model in enumerate(sorted(self.models_paths)):

            if verbose:
                print("Reading model: %s" % model)

            if collect_memory_every and i % collect_memory_every == 0:
                collect_memory = True
            else:
                collect_memory = False

            self.models_names.append(model)
            self.readModelFromPDB(
                model,
                self.models_paths[model],
                add_to_path=True,
                collect_memory=collect_memory,
                only_hetatoms=only_hetatm_conects,
                wat_to_hoh=wat_to_hoh,
            )

        if get_sequences:
            # Get sequence information based on stored structure objects
            self.getModelsSequences()

        if get_ss:
            # Calculate secondary structure inforamtion as strings
            self.calculateSecondaryStructure()

        # # Perform a multiple sequence aligment of models
        if msa:
            if self.multichain:
                print(
                    "MSA cannot be calculated at startup when multichain models \
are given. See the calculateMSA() method for selecting which chains will be algined."
                )
            else:
                self.calculateMSA()

    def addResidueToModel(
        self,
        model,
        chain_id,
        resname,
        atom_names,
        coordinates,
        new_resid=None,
        elements=None,
        hetatom=True,
        water=False,
    ):
        """
        Add a residue to a specific model.

        Parameters
        ==========
        model : str
            Model name to edit
        chain_id : str
            Chain ID to which the residue will be added.
        resname : str
            Name of the residue to be added.
        atom_names : list ot tuple
            Atom names of each atom in the residue to add.
        coordinates : numpy.ndarray
            Atom coordinates array, it should match the order in the given
            atom_names.
        elements : list
            List of atomic elements. One per each atom.
        hetatom : bool
            Is the residue an hetatm?
        water : bool
            Is the residue a water residue?
        """

        # Check model name
        if model not in self.structures:
            raise ValueError("The input model was not found.")

        # Check chain ID
        chain = [
            chain
            for chain in self.structures[model].get_chains()
            if chain_id == chain.id
        ]
        if len(chain) != 1:
            print("Chain ID %s was not found in the selected model." % chain_id)
            print("Creating a new chain with ID %s" % chain_id)
            new_chain = PDB.Chain.Chain(chain_id)
            for m in self.structures[model]:
                m.add(new_chain)
            chain = [
                chain
                for chain in self.structures[model].get_chains()
                if chain_id == chain.id
            ]

        # Check coordinates correctness
        if coordinates.shape == ():
            if np.isnan(coordinates):
                raise ValueError("Given Coordinate in nan!")
        elif np.isnan(coordinates.any()):
            raise ValueError("Some given Coordinates are nan!")
        if coordinates.shape[1] != 3:
            raise ValueError(
                "Coordinates must have shape (x,3). X=number of atoms in residue."
            )
        if len(coordinates.shape) > 1:
            if coordinates.shape[0] != len(atom_names):
                raise ValueError(
                    "Mismatch between the number of atom_names and coordinates."
                )
        if len(coordinates.shape) == 1:
            if len(atom_names) != 1:
                raise ValueError(
                    "Mismatch between the number of atom_names and coordinates."
                )

        # Create new residue
        if new_resid == None:
            try:
                new_resid = max([r.id[1] for r in chain[0].get_residues()]) + 1
            except:
                new_resid = 1

        rt_flag = " "  # Define the residue type flag for complete the residue ID.
        if hetatom:
            rt_flag = "H"
        if water:
            rt_flag = "W"
        residue = PDB.Residue.Residue((rt_flag, new_resid, " "), resname, " ")

        # Add new atoms to residue
        try:
            serial_number = max([a.serial_number for a in chain[0].get_atoms()]) + 1
        except:
            serial_number = 1
        for i, atnm in enumerate(atom_names):
            if elements:
                atom = PDB.Atom.Atom(
                    atom_names[i],
                    coordinates[i],
                    0,
                    1.0,
                    " ",
                    "%-4s" % atom_names[i],
                    serial_number + i,
                    elements[i],
                )
            else:
                atom = PDB.Atom.Atom(
                    atom_names[i],
                    coordinates[i],
                    0,
                    1.0,
                    " ",
                    "%-4s" % atom_names[i],
                    serial_number + i,
                )
            residue.add(atom)
        chain[0].add(residue)

        return new_resid

    def _removeAtomsFromConects(self, model, atom_keys):
        """
        Remove CONECT entries referencing any atom tuples in ``atom_keys``.
        """
        if not atom_keys:
            return
        conects = self.conects.get(model)
        if not conects:
            return
        atom_key_set = set(atom_keys)
        self.conects[model] = [
            conect
            for conect in conects
            if atom_key_set.isdisjoint(conect)
        ]

    def removeModelAtoms(self, model, atoms_list, verbose=True):
        """
        Remove specific atoms of a model. Atoms to delete are given as a list of tuples.
        Each tuple contains three positions specifying (chain_id, residue_id, atom_name).

        Paramters
        =========
        model : str
            model ID
        atom_lists : list
            Specifies the list of atoms to delete for the particular model.
        """

        removed_atoms = []
        for remove_atom in atoms_list:
            for chain in self.structures[model].get_chains():
                if chain.id == remove_atom[0]:
                    for residue in chain:
                        if residue.id[1] == remove_atom[1]:
                            for atom in residue:
                                if atom.name == remove_atom[2]:
                                    if verbose:
                                        print(
                                            "Removing atom: "
                                            + str(remove_atom)
                                            + " from model "
                                            + model
                                        )
                                    residue.detach_child(atom.id)
                                    removed_atoms.append(remove_atom)
        if removed_atoms:
            self._removeAtomsFromConects(model, removed_atoms)

    def removeModelResidues(self, model, residues_list, verbose=True):
        """
        Remove a group of residues from the model structure.

        Parameters
        ==========
        model : str
            Identifier of the model whose residues should be removed. Must exist in
            `self.structures`.
        residues_list : iterable
            Residue selectors describing the residues to delete. Each entry can be:

              * a ``Bio.PDB.Residue`` instance obtained from the structure;
              * a tuple ``(chain_id, resseq)`` (legacy behaviour);
              * a tuple ``(chain_id, residue_id_tuple)`` where ``residue_id_tuple`` is
                the three-element Biopython ``Residue.id`` (``hetfield``, ``resseq``,
                ``icode``); or
              * a bare ``Residue.id`` tuple (three elements) to match residues across
                any chain. When such tuples appear multiple times their multiplicity is
                respected (e.g. duplicating the tuple removes that many matching
                residues).
        verbose : bool, optional
            If True (default) log each residue removal.

        Raises
        ======
        ValueError
            Raised when no residues matching the provided selectors are found.
        TypeError
            Raised when selectors cannot be interpreted.
        """

        if model not in self.structures:
            raise ValueError(f"Model '{model}' is not present in the current session.")

        try:
            requested = list(residues_list)
        except TypeError as exc:
            raise TypeError(
                "residues_list must be an iterable of residue identifiers."
            ) from exc

        if not requested:
            raise ValueError("No residue identifiers were supplied.")

        structure = self.structures[model]

        explicit_full = Counter()
        explicit_resseq = Counter()
        global_full = Counter()
        explicit_full_desc = {}
        explicit_resseq_desc = {}
        global_full_desc = {}

        def _coerce_resseq(value):
            if isinstance(value, int):
                return value
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Residue sequence identifier {value!r} is not an integer."
                ) from exc

        def _check_residue_id(residue_id):
            if (
                not isinstance(residue_id, tuple)
                or len(residue_id) != 3
                or not isinstance(residue_id[1], int)
            ):
                raise TypeError(
                    f"Residue identifier {residue_id!r} is not a valid Biopython Residue.id tuple."
                )

        for entry in requested:
            if isinstance(entry, PDB.Residue.Residue):
                chain_id = entry.get_parent().id
                residue_id = entry.id
                explicit_full[(chain_id, residue_id)] += 1
                explicit_full_desc.setdefault(
                    (chain_id, residue_id), f"{chain_id}:{residue_id}"
                )
                continue

            if not isinstance(entry, tuple):
                raise TypeError(
                    "Residue selectors must be Bio.PDB.Residue objects or tuples."
                )

            if len(entry) == 2:
                chain_id, residue_token = entry
                if isinstance(residue_token, tuple):
                    _check_residue_id(residue_token)
                    explicit_full[(chain_id, residue_token)] += 1
                    explicit_full_desc.setdefault(
                        (chain_id, residue_token), f"{chain_id}:{residue_token}"
                    )
                else:
                    resseq = _coerce_resseq(residue_token)
                    explicit_resseq[(chain_id, resseq)] += 1
                    explicit_resseq_desc.setdefault(
                        (chain_id, resseq), f"{chain_id}:resseq={resseq}"
                    )
                continue

            if len(entry) == 3:
                _check_residue_id(entry)
                global_full[entry] += 1
                global_full_desc.setdefault(entry, f"{entry}")
                continue

            if len(entry) == 4:
                chain_id = entry[0]
                residue_id = tuple(entry[1:])
                _check_residue_id(residue_id)
                explicit_full[(chain_id, residue_id)] += 1
                explicit_full_desc.setdefault(
                    (chain_id, residue_id), f"{chain_id}:{residue_id}"
                )
                continue

            raise TypeError(f"Unsupported residue selector format: {entry!r}")

        matched_explicit_full = Counter()
        matched_explicit_resseq = Counter()
        matched_global_full = Counter()

        atoms_to_remove = []
        residues_to_detach = {}

        for chain in structure.get_chains():
            chain_id = chain.id
            for residue in chain:
                full_key = (chain_id, residue.id)
                resseq_key = (chain_id, residue.id[1])
                matched = False

                if explicit_full[full_key] > matched_explicit_full[full_key]:
                    matched_explicit_full[full_key] += 1
                    matched = True
                elif explicit_resseq[resseq_key] > matched_explicit_resseq[resseq_key]:
                    matched_explicit_resseq[resseq_key] += 1
                    matched = True
                elif global_full[residue.id] > matched_global_full[residue.id]:
                    matched_global_full[residue.id] += 1
                    matched = True

                if matched:
                    residues_to_detach.setdefault((chain_id, residue.id), chain)
                    for atom in residue:
                        atoms_to_remove.append((chain_id, residue.id[1], atom.name))

        missing = []
        for key, count in explicit_full.items():
            deficit = count - matched_explicit_full[key]
            if deficit > 0:
                missing.extend([explicit_full_desc[key]] * deficit)
        for key, count in explicit_resseq.items():
            deficit = count - matched_explicit_resseq[key]
            if deficit > 0:
                missing.extend([explicit_resseq_desc[key]] * deficit)
        for key, count in global_full.items():
            deficit = count - matched_global_full[key]
            if deficit > 0:
                missing.extend([global_full_desc[key]] * deficit)

        if not residues_to_detach:
            if missing:
                raise ValueError(
                    "None of the requested residues were found: "
                    + ", ".join(sorted(set(missing)))
                )
            raise ValueError("No atoms were found for the specified residue!")

        if missing:
            warnings.warn(
                "Some requested residues were not found and were skipped: "
                + ", ".join(sorted(set(missing))),
                UserWarning,
            )

        self._removeAtomsFromConects(model, atoms_to_remove)

        for (chain_id, residue_id), chain_obj in residues_to_detach.items():
            if verbose:
                resseq = residue_id[1]
                icode = residue_id[2].strip() if isinstance(residue_id[2], str) else residue_id[2]
                if icode:
                    print(
                        f"Removing residue: ({chain_id}, {resseq}{icode}) from model {model}"
                    )
                else:
                    print(
                        f"Removing residue: ({chain_id}, {resseq}) from model {model}"
                    )
            if residue_id in chain_obj.child_dict:
                chain_obj.detach_child(residue_id)

    def removeModelChains(self, model, chains, verbose=True):
        """
        Remove one or more chains from a stored model.

        Parameters
        ==========
        model : str
            Identifier of the model whose chains should be removed. Must exist in
            ``self.structures``.
        chains : str, Bio.PDB.Chain.Chain, or iterable
            Chain selectors. Each entry may be a chain ID string or a Biopython
            ``Chain`` object. Iterables are flattened one level (e.g. list or set of
            chain IDs). Duplicate IDs are ignored after the first occurrence.
        verbose : bool, optional
            If True (default) log each chain removal.

        Raises
        ======
        ValueError
            Raised when the model is unknown, no chains are provided, or none of the
            requested chains are present in the model.
        TypeError
            Raised when chain selectors cannot be interpreted.
        """

        if model not in self.structures:
            raise ValueError(f"Model '{model}' is not present in the current session.")

        if isinstance(chains, (str, PDB.Chain.Chain)):
            selector_entries = [chains]
        else:
            try:
                selector_entries = list(chains)
            except TypeError as exc:
                raise TypeError("chains must be a chain ID, Chain object, or iterable.") from exc

        if not selector_entries:
            raise ValueError("No chain identifiers were supplied.")

        chain_id_order = []
        for entry in selector_entries:
            if isinstance(entry, PDB.Chain.Chain):
                chain_id = entry.id
            elif isinstance(entry, str):
                if not entry:
                    raise ValueError("Chain identifier strings must be non-empty.")
                chain_id = entry
            else:
                raise TypeError(
                    "Chain selectors must be strings or Bio.PDB.Chain.Chain objects."
                )
            chain_id_order.append(chain_id)

        # Preserve user order while removing duplicates
        seen = set()
        target_chain_ids = []
        for cid in chain_id_order:
            if cid not in seen:
                seen.add(cid)
                target_chain_ids.append(cid)

        target_chain_set = set(target_chain_ids)

        structure = self.structures[model]
        models_list = list(structure.get_models())

        atoms_to_remove = []
        chains_to_detach = []
        found_chain_ids = set()

        for stored_model in models_list:
            for chain in list(stored_model.get_chains()):
                if chain.id not in target_chain_set:
                    continue
                found_chain_ids.add(chain.id)
                chains_to_detach.append((stored_model, chain))
                for residue in chain.get_residues():
                    for atom in residue:
                        atoms_to_remove.append((chain.id, residue.id[1], atom.name))

        if not found_chain_ids:
            raise ValueError(
                "No chains matching the provided identifiers were found in model "
                f"'{model}'."
            )

        missing = [cid for cid in target_chain_ids if cid not in found_chain_ids]
        self._removeAtomsFromConects(model, atoms_to_remove)

        for stored_model, chain_obj in chains_to_detach:
            if verbose:
                print(f"Removing chain: {chain_obj.id} from model {model}")
            if chain_obj.id in stored_model.child_dict:
                stored_model.detach_child(chain_obj.id)

        if missing:
            warnings.warn(
                "Some requested chains were not found and were skipped: "
                + ", ".join(sorted(set(missing))),
                UserWarning,
            )

    def changeResidueAtomNames(
        self,
        residue_name,
        atom_names,
        models=None,
        verbose=False,
        summary=False,
    ):
        """
        Rename atoms for residues matching ``residue_name`` across stored models.

        Parameters
        ----------
        residue_name : str
            Residue name (e.g. ligand three-letter code) to target.
        atom_names : dict
            Mapping from original four-character (including padding) atom names to
            their four-character replacements.
        models : iterable or str, optional
            Specific model name(s) to modify. Defaults to all loaded models.
        verbose : bool, optional
            Print each replacement as it occurs.
        summary : bool, optional
            Print a summary of replacements once processing finishes.

        Returns
        -------
        int
            Total number of atom names updated.
        """

        if not isinstance(residue_name, str) or not residue_name.strip():
            raise ValueError("residue_name must be a non-empty string.")

        if not isinstance(atom_names, dict) or not atom_names:
            raise ValueError("atom_names must be a non-empty dictionary.")

        normalized_map = {}
        found_map = {}
        for old_name, new_name in atom_names.items():
            if not isinstance(old_name, str) or not isinstance(new_name, str):
                raise ValueError("Atom name mappings must use strings as keys and values.")

            if len(old_name) != 4:
                raise ValueError(
                    f'Atom name mapping keys must be exactly four characters long (including padding); '
                    f'received "{old_name}".'
                )
            if len(new_name) != 4:
                raise ValueError(
                    f'Atom name mapping values must be exactly four characters long (including padding); '
                    f'received "{new_name}".'
                )

            stripped_old = old_name.strip()
            stripped_new = new_name.strip()

            if not stripped_old:
                raise ValueError("Atom name mapping keys cannot be empty.")
            if not stripped_new:
                raise ValueError("Atom name mapping values cannot be empty.")

            if stripped_old in normalized_map and normalized_map[stripped_old] != stripped_new:
                raise ValueError(
                    f'Conflicting mappings supplied for atom "{stripped_old}".'
                )

            normalized_map[stripped_old] = stripped_new
            found_map[stripped_old] = False

        if isinstance(models, str):
            target_models = [models]
        elif models is None:
            target_models = list(self.structures.keys())
        else:
            target_models = list(models)

        if not target_models:
            return 0

        target_models = list(dict.fromkeys(target_models))

        missing_models = [m for m in target_models if m not in self.structures]
        if missing_models:
            raise KeyError(
                "Requested models are not loaded: %s" % ", ".join(sorted(missing_models))
            )

        total_replacements = 0
        summary_records = {}
        target_resname = residue_name.strip().upper()

        for model_name in target_models:
            structure = self.structures[model_name]
            for stored_model in structure.get_models():
                for chain in stored_model.get_chains():
                    for residue in chain.get_residues():
                        if residue.get_resname().strip().upper() != target_resname:
                            continue

                        residue_id = residue.id[1]

                        atoms = list(residue.get_atoms())
                        atoms_by_name = {a.get_name().strip(): a for a in atoms}
                        target_names = [
                            old_name for old_name in normalized_map if old_name in atoms_by_name
                        ]
                        if not target_names:
                            continue

                        residue_child_dict = getattr(residue, "child_dict", None)
                        if residue_child_dict is None:
                            raise ValueError(
                                "Residue object is missing child_dict; cannot rename atoms."
                            )

                        non_target_names = set(atoms_by_name).difference(target_names)
                        for old_name in target_names:
                            new_atom_name = normalized_map[old_name]
                            if (
                                new_atom_name in non_target_names
                                and new_atom_name not in target_names
                            ):
                                raise ValueError(
                                    f'Renaming atom "{old_name}" to "{new_atom_name}" '
                                    f'would overwrite an existing atom in residue {chain.id}:{residue_id} '
                                    f'of model "{model_name}".'
                                )

                        new_names_for_targets = [normalized_map[name] for name in target_names]
                        if len(new_names_for_targets) != len(set(new_names_for_targets)):
                            raise ValueError(
                                f"Atom name mapping produces duplicates for residue {chain.id}:{residue_id} "
                                f'of model "{model_name}".'
                            )

                        reserved_names = set(atoms_by_name)
                        reserved_names.update(normalized_map.values())

                        temp_assignments = {}
                        residue_name_updates = {}

                        def _generate_temp_name(counter: int) -> str:
                            attempt = counter
                            while True:
                                candidate = f"T{attempt:03d}"
                                if candidate not in reserved_names:
                                    reserved_names.add(candidate)
                                    return candidate
                                attempt += 1

                        # First pass: move targeted atoms to unique temporary names to avoid clashes
                        for idx, old_name in enumerate(target_names):
                            atom = atoms_by_name[old_name]
                            temp_name = _generate_temp_name(idx)

                            residue_child_dict.pop(atom.id, None)
                            atom.name = temp_name
                            atom.id = temp_name
                            atom.fullname = f"{temp_name:>4s}"
                            atom.full_id = atom.get_full_id()
                            residue_child_dict[temp_name] = atom

                            temp_assignments[old_name] = (atom, temp_name)
                            atoms_by_name[temp_name] = atom
                            atoms_by_name.pop(old_name, None)

                        # Second pass: assign the requested final names
                        for old_name in target_names:
                            atom, temp_name = temp_assignments[old_name]
                            new_atom_name = normalized_map[old_name]

                            residue_child_dict.pop(temp_name, None)
                            if verbose:
                                print(
                                    f'[{model_name}] chain {chain.id} residue {residue_id}: '
                                    f'{old_name} -> {new_atom_name}'
                                )

                            atom.name = new_atom_name
                            atom.id = new_atom_name
                            atom.fullname = f"{new_atom_name:>4s}"
                            atom.full_id = atom.get_full_id()
                            residue_child_dict[new_atom_name] = atom

                            atoms_by_name[new_atom_name] = atom
                            atoms_by_name.pop(temp_name, None)

                            found_map[old_name] = True
                            total_replacements += 1
                            residue_name_updates[old_name] = new_atom_name
                            if summary:
                                key = (model_name, chain.id, residue_id)
                                summary_records[key] = summary_records.get(key, 0) + 1

                        if residue_name_updates and model_name in self.conects:
                            updated_conects = []
                            for conect in self.conects[model_name]:
                                updated_conect = []
                                for entry_chain, entry_resid, entry_atom in conect:
                                    if (
                                        entry_chain == chain.id
                                        and entry_resid == residue_id
                                        and entry_atom in residue_name_updates
                                    ):
                                        updated_conect.append(
                                            (
                                                entry_chain,
                                                entry_resid,
                                                residue_name_updates[entry_atom],
                                            )
                                        )
                                    else:
                                        updated_conect.append(
                                            (entry_chain, entry_resid, entry_atom)
                                        )
                                updated_conects.append(updated_conect)
                            self.conects[model_name] = updated_conects

        for atom_key, was_found in found_map.items():
            if not was_found:
                print(
                    f'Atom name "{atom_key}" was not found in residues named "{target_resname}".'
                )

        if summary:
            if total_replacements:
                print(
                    f'Renamed {total_replacements} atom(s) across {len(summary_records)} residue(s).'
                )
                for key, count in sorted(summary_records.items()):
                    model_name, chain_id, residue_id = key
                    print(
                        f'  [{model_name}] chain {chain_id} residue {residue_id}: {count} atom(s) renamed.'
                    )
            else:
                print("No atom names were updated.")

        return total_replacements

    def removeAtomFromConectLines(self, residue_name, atom_name, verbose=True):
        """
        Remove the given (atom_name) atoms from all the connect lines involving
        the given (residue_name) residues.
        """

        # Match all the atoms with the given residue and atom name

        for model in self:
            resnames = {}
            for chain in self.structures[model].get_chains():
                for residue in chain:
                    resnames[(chain.id, residue.id[1])] = residue.resname

            conects = []
            count = 0
            for conect in self.conects[model]:
                new_conect = []
                for atom in conect:
                    if resnames[atom[:-1]] != residue_name and atom[-1] != atom_name:
                        new_conect.append(atom)
                    else:
                        count += 1
                if new_conect == []:
                    continue
                conects.append(new_conect)
            self.conects[model] = conects
            if verbose:
                print(f"Removed {count} from conect lines of model {model}")

    def addCappingGroups(self, rosetta_style_caps=False, prepwizard_style_caps=False,
                         openmm_style_caps=False, stdout=False, stderr=False,
                         conect_update=True, only_hetatoms=False):

        if sum([bool(rosetta_style_caps), bool(prepwizard_style_caps), bool(openmm_style_caps)]) > 1:
            raise ValueError('You must give only on cap style option!')

        # Manage stdout and stderr
        if stdout:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if stderr:
            stderr = None
        else:
            stderr = subprocess.DEVNULL

        if not os.path.exists('_capping_'):
            os.mkdir('_capping_')

        if not os.path.exists('_capping_/input_models'):
            os.mkdir('_capping_/input_models')

        if not os.path.exists('_capping_/output_models'):
            os.mkdir('_capping_/output_models')

        self.saveModels('_capping_/input_models')

        _copyScriptFile('_capping_', "addCappingGroups.py")
        command =  'run python3 _capping_/._addCappingGroups.py '
        command += '_capping_/input_models/ '
        command += '_capping_/output_models/ '
        if rosetta_style_caps:
            command += '--rosetta_style_caps '
        elif prepwizard_style_caps:
            command += '--prepwizard_style_caps '
        elif openmm_style_caps:
            command += '--openmm_style_caps '

        subprocess.run(command, shell=True, stdout=stdout, stderr=stderr)

        for f in os.listdir('_capping_/output_models'):
            model = f.replace('.pdb', '')
            self.readModelFromPDB(model, '_capping_/output_models/'+f, conect_update=conect_update, only_hetatoms=only_hetatoms)
        shutil.rmtree('_capping_')

    def removeCaps(self, models=None, remove_ace=True, remove_nma=True):
        """
        Remove caps from models.
        """

        for model in self:

            if models and model not in models:
                continue

            for chain in self.structures[model].get_chains():

                st_residues = [r for r in chain if r.resname in aa3]

                ACE = None
                NMA = None
                NT = None
                CT = None

                for residue in self.structures[model].get_residues():
                    if residue.resname == "ACE":
                        ACE = residue
                    elif residue.resname == "NMA":
                        NMA = residue

                for i, residue in enumerate(chain):

                    if (
                        ACE
                        and residue.id[1] == ACE.id[1] + 1
                        and residue.resname in aa3
                    ):
                        NT = residue
                    elif not ACE and i == 0:
                        NT = residue
                    if NMA and residue.id[1] == NMA.id[1] and residue.resname in aa3:
                        CT = residue
                    elif not NMA and i == len(st_residues) - 1:
                        CT = residue

                # Remove termini
                if ACE and remove_ace:
                    for a in ACE:
                        self.removeAtomFromConectLines("ACE", a.name, verbose=False)
                    ACE.get_parent().detach_child(ACE.id)

                if NMA and remove_nma:
                    for a in NMA:
                        self.removeAtomFromConectLines("NMA", a.name, verbose=False)
                    NMA.get_parent().detach_child(NMA.id)

    def addOXTAtoms(self):
        """
        Add missing OXT atoms for terminal residues when missing
        """

        # Define internal coordinates for OXT
        oxt_c_distance = 1.251
        oxt_c_o_angle = 124.222
        oxt_c_o_ca = 179.489

        for model in self:
            for chain in self.structures[model].get_chains():
                residues = [r for r in chain if r.id[0] == " "]
                if residues == []:
                    continue
                last_atoms = {a.name: a for a in residues[-1]}

                if "OXT" not in last_atoms:

                    oxt_coord = _computeCartesianFromInternal(
                        last_atoms["C"].coord,
                        last_atoms["O"].coord,
                        last_atoms["CA"].coord,
                        1.251,
                        124.222,
                        179.489,
                    )
                    serial_number = max([a.serial_number for a in residues[-1]]) + 1
                    oxt = PDB.Atom.Atom(
                        "OXT", oxt_coord, 0, 1.0, " ", "OXT ", serial_number + 1, "O"
                    )
                    residues[-1].add(oxt)

    def readModelFromPDB(
        self,
        model,
        pdb_file,
        wat_to_hoh=False,
        covalent_check=True,
        atom_mapping=None,
        add_to_path=False,
        conect_update=True,
        collect_memory=False,
        only_hetatoms=False,
    ):
        """
        Adds a model from a PDB file.

        Parameters
        ----------
        model : str
            Model name.
        pdb_file : str
            Path to the pdb file.
        wat_to_hoh : bool
            Change the water name from WAT to HOH. Specially useful when resing from
            Rosetta optimization output files containing water.

        Returns
        -------
        structure : Bio.PDB.Structure
            Structure object.
        """

        if model not in self.models_names:
            self.models_names.append(model)

        self.structures[model] = _readPDB(model, pdb_file)

        if wat_to_hoh:
            for residue in self.structures[model].get_residues():
                if residue.resname == "WAT":
                    residue.resname = "HOH"

        # Read conect lines (avoid double read if covalent_check may rewrite PDB)
        if conect_update and not covalent_check:
            if model not in self.conects or self.conects[model] == []:
                self.conects[model] = self._readPDBConectLines(
                    pdb_file, model, only_hetatoms=only_hetatoms
                )

        # Check covalent ligands
        if covalent_check:
            self._checkCovalentLigands(model, pdb_file, atom_mapping=atom_mapping)

        # Update conect lines once after potential covalent sorting
        if conect_update:
            self.conects[model] = self._readPDBConectLines(
                pdb_file, model, only_hetatoms=only_hetatoms
            )

        if add_to_path:
            self.models_paths[model] = pdb_file

        if collect_memory:
            gc.collect()  # Collect memory

        return self.structures[model]

    def getModelsSequences(self):
        """
        Get sequence information for all stored models. It modifies the self.multi_chain
        option to True if more than one chain is found in the models.

        Returns
        =======
        sequences : dict
            Contains the sequences of all models.
        """
        self.multi_chain = False
        self.chain_sequences = {}

        for model in self.models_names:
            chains = [c for c in self.structures[model].get_chains()]
            per_chain_sequences = {}
            for c in chains:
                per_chain_sequences[c.id] = self._getChainSequence(c)

            self.chain_sequences[model] = per_chain_sequences

            if len(per_chain_sequences) == 1:
                self.sequences[model] = next(iter(per_chain_sequences.values()))
            else:
                self.sequences[model] = per_chain_sequences

            valid_sequences = [
                seq for seq in per_chain_sequences.values() if seq not in (None, "")
            ]
            if len(valid_sequences) > 1:
                # If any model has more than one polymeric chain set multi_chain to True.
                self.multi_chain = True

        return self.sequences

    def _resolve_chain_selection(self, chains=None, models=None, require_single=False):
        """
        Normalize user-provided chain selections.

        Parameters
        ----------
        chains : None, str, iterable, or dict
            Chain selector. When dict, keys are model names and values can be single
            chain IDs or iterables of chain IDs.
        models : iterable, optional
            Subset of models to consider. Defaults to all loaded models.
        require_single : bool, optional
            When True, ensure a single chain ID is returned per model.

        Returns
        -------
        dict
            Mapping of model -> list of chain IDs (require_single=False) or
            model -> single chain ID (require_single=True).
        """
        if models is None:
            models = self.models_names

        # Ensure sequence information is populated
        if self.chain_sequences == {} and self.structures:
            self.getModelsSequences()

        selection = {}

        def _normalize_value(value):
            if value is None:
                return None
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple, set)):
                return list(dict.fromkeys(value))
            raise TypeError(
                "Chain selectors must be strings, iterables, or dictionaries mapping to them."
            )

        for model in models:
            available_sequences = self.chain_sequences.get(model, {})
            polymer_chains = [
                chain_id
                for chain_id, seq in available_sequences.items()
                if seq not in (None, "")
            ]
            if polymer_chains:
                available = list(polymer_chains)
            elif available_sequences:
                available = list(available_sequences.keys())
            else:
                available = [c.id for c in self.structures[model].get_chains()]

            if isinstance(chains, dict):
                requested = _normalize_value(chains.get(model))
            else:
                requested = _normalize_value(chains)

            if requested is None:
                requested = available

            missing = [c for c in requested if c not in available]
            if missing:
                raise ValueError(
                    f"Requested chain(s) {missing} not found in model {model}. "
                    f"Available chains: {available}"
                )

            if require_single:
                if len(requested) != 1:
                    raise ValueError(
                        f"Model {model} requires a single chain selection but received {requested}."
                    )
                selection[model] = requested[0]
            else:
                selection[model] = requested

        return selection

    def renumberModels(self, by_chain=True):
        """
        Renumber every PDB chain residues from 1 onward.
        """

        for m in self:
            residue_mapping = {}
            structure = PDB.Structure.Structure(0)
            pdb_model = PDB.Model.Model(0)
            for model in self.structures[m]:
                i = 0
                for c in model:
                    if by_chain:
                        i = 0
                    residues = [r for r in c]
                    chain_copy = PDB.Chain.Chain(c.id)
                    for r in residues:
                        chain_id = c.id
                        old_resid = r.id[1]
                        new_id = (r.id[0], i + 1, r.id[2])
                        c.detach_child(r.id)
                        r.id = new_id
                        chain_copy.add(r)
                        residue_mapping[(chain_id, old_resid)] = new_id[1]
                        i += 1
                    pdb_model.add(chain_copy)

            structure.add(pdb_model)
            self.structures[m] = structure
            if m in self.conects and residue_mapping:
                updated_conects = []
                for conect in self.conects[m]:
                    updated_conect = []
                    for chain_id, resid, atom_name in conect:
                        new_resid = residue_mapping.get((chain_id, resid), resid)
                        updated_conect.append((chain_id, new_resid, atom_name))
                    updated_conects.append(updated_conect)
                self.conects[m] = updated_conects

    def calculateMSA(self, extra_sequences=None, chains=None):
        """
        Calculate a Multiple Sequence Alignment from the current models' sequences.

        Returns
        =======
        alignment : Bio.AlignIO
            Multiple sequence alignment in Biopython format.
        chains : dict
            Dictionary specifying which chain to use for each model
        """

        # Normalize chain selection (requires a single chain per model)
        chain_selection = self._resolve_chain_selection(
            chains, require_single=True
        )

        sequences = {}
        for model in self.models_names:
            chain_id = chain_selection.get(model)
            if chain_id is None:
                continue

            seq = self.chain_sequences.get(model, {}).get(chain_id)
            if seq in (None, ""):
                raise ValueError(
                    f"Selected chain {chain_id} in model {model} does not contain a protein sequence."
                )

            sequences[model] = seq

        if isinstance(extra_sequences, dict):
            sequences.update(extra_sequences)

        self.msa = alignment.mafft.multipleSequenceAlignment(sequences, stderr=False)
        self.msa_chain_selection = chain_selection

        return self.msa

    def getConservedMSAPositions(self, msa):
        """
        Get all conserved MSA positions.

        Returns
        =======
        conserved : list
            All conserved MSA positions indexes and the conserved amino acid identity.
        """

        positions = {}
        conserved = []
        n_models = len(self.msa)
        for i in range(self.msa.get_alignment_length()):
            positions[i] = []
            for model in self.msa:
                positions[i].append(model.seq[i])
            positions[i] = set(positions[i])
            if len(positions[i]) == 1:
                conserved.append((i, list(positions[i])[0]))

        return conserved

    def getStructurePositionsFromMSAindexes(self, msa_indexes, msa=None, models=None):
        """
        Get the individual model residue structure positions of a set of MSA indexes
        Paramters
        =========
        msa_indexes : list
            Zero-based MSA indexes
        Returns
        =======
        residue_indexes : dict
            Residue indexes for each protein at the MSA positions
        """

        if isinstance(msa_indexes, int):
            msa_indexes = [msa_indexes]

        if models == None:
            models = []
        elif isinstance(models, str):
            models = [models]

        # If msa not given get the class msa attribute
        if msa == None:
            msa = self.msa

        positions = {}
        residue_ids = {}

        # Gather dictionary between sequence position and residue PDB index
        for model in self.models_names:
            if models != [] and model not in models:
                continue

            positions[model] = 0
            residue_ids[model] = {}
            for i, r in enumerate(self.structures[model].get_residues()):
                residue_ids[model][i + 1] = r.id[1]

        # Gather sequence indexes for the given MSA index
        sequence_positions = {}
        for i in range(msa.get_alignment_length()):
            # Count structure positions
            for entry in msa:

                if entry.id not in self.models_names:
                    continue
                sequence_positions.setdefault(entry.id, [])

                if entry.seq[i] != "-":
                    positions[entry.id] += 1

            # Get residue positions matching the MSA indexes
            if i in msa_indexes:
                for entry in msa:
                    if entry.id not in self.models_names:
                        continue

                    if entry.seq[i] == "-":
                        sequence_positions[entry.id].append(None)
                    else:
                        sequence_positions[entry.id].append(
                            residue_ids[entry.id][positions[entry.id]]
                        )

        return sequence_positions

    import mdtraj as md

    def calculateSecondaryStructure(self, simplified=True, chains=None, _save_structure=False):
        """
        Calculate secondary structure information for each model using MDTraj.

        Parameters
        ==========
        simplified : bool, default=False
            If True, reduces the DSSP codes to:
            - H (helix) → "H"
            - E (sheet) → "E"
            - Everything else → "C" (coil)
        chains : None, str, iterable or dict, optional
            Chain selector to limit the calculation. When None, all chains per
            model are processed.
        _save_structure : bool, optional
            Unused legacy argument retained for backwards compatibility.

        frame : int, default=0
            Frame index to extract the secondary structure from (MD simulations).

        dssp : str, default='score'
            The DSSP algorithm to use. Options:
            - 'score' : Uses MDTraj’s built-in DSSP scoring method.
            - 'sander' : Uses Sander DSSP method.
            - 'mkdssp' : Calls external DSSP executable (if available).

        Returns
        =======
        ss : dict
            Contains the secondary structure strings for each model.
        """

        chain_selection = self._resolve_chain_selection(
            chains, require_single=False
        )

        self.ss = {}
        self.chain_secondary_structure = {}

        for model in self.models_names:

            # Load structure into MDTraj
            traj = md.load(self.models_paths[model])

            # Compute secondary structure
            assignments = md.compute_dssp(traj, simplified=simplified)[0]
            per_chain_ss = {}

            for residue, code in zip(traj.topology.residues, assignments):
                chain_id = residue.chain.chain_id
                per_chain_ss.setdefault(chain_id, []).append(code)

            normalized = {}
            for chain_id in chain_selection.get(model, []):
                codes = per_chain_ss.get(chain_id, [])
                normalized[chain_id] = "".join(codes)

            # Keep a record for downstream multi-chain aware operations
            self.chain_secondary_structure[model] = normalized

            if len(normalized) == 1:
                self.ss[model] = next(iter(normalized.values()))
            else:
                self.ss[model] = normalized

        return self.ss

    def keepModelChains(self, model, chains):
        """
        Only keep the specified chains for the selected model.

        Parameters
        ==========
        model : str
            Model name
        chains : list or tuple or str
            Chain IDs to keep.
        """
        if isinstance(chains, str):
            chains = list(chains)

        remove = []
        for chain in self.structures[model].get_chains():
            if chain.id not in chains:
                print("From model %s Removing chain %s" % (model, chain.id))
                remove.append(chain)

        model = [*self.structures[model].get_models()][0]
        for chain in remove:
            model.detach_child(chain.id)

        self.getModelsSequences()

    def removeTerminalUnstructuredRegions(self, n_hanging=3, chains=None, verbose=False):
        """
        Remove unstructured terminal regions from models.

        Parameters
        ==========
        n_hangin : int
            Maximum unstructured number of residues to keep at the unstructured terminal regions.
        chains : None, str, iterable or dict
            Chain selector identifying which chains should be processed.
        verbose : bool, default=False
            When True, log the residues that are trimmed per model/chain.
        """

        if not self.chain_secondary_structure:
            self.calculateSecondaryStructure()

        chain_selection = self._resolve_chain_selection(
            chains, require_single=False
        )

        def _resolve_n_hanging(model, chain_id):
            if isinstance(n_hanging, dict):
                model_value = n_hanging.get(model)
                if isinstance(model_value, dict):
                    if chain_id in model_value:
                        return model_value[chain_id]
                elif isinstance(model_value, int):
                    return model_value

                chain_value = n_hanging.get(chain_id)
                if isinstance(chain_value, int):
                    return chain_value

                default_value = n_hanging.get("default")
                if isinstance(default_value, int):
                    return default_value

                raise ValueError(
                    f"Could not resolve n_hanging for model {model} chain {chain_id}."
                )
            return int(n_hanging)

        def _is_unstructured(code):
            return code not in ("H", "E")

        # Calculate residues to be removed
        for model in self.models_names:

            model_selection = chain_selection.get(model, [])
            model_ss = self.chain_secondary_structure.get(model, {})
            if not model_selection:
                continue

            # Fetch chain objects once for reuse
            chains_map = {c.id: c for c in self.structures[model].get_chains()}

            for chain_id in model_selection:
                chain_obj = chains_map.get(chain_id)
                if chain_obj is None:
                    continue

                ss_string = model_ss.get(chain_id, "")
                if not ss_string:
                    continue

                limit = _resolve_n_hanging(model, chain_id)

                # N-terminus
                leading = []
                for idx, code in enumerate(ss_string):
                    if _is_unstructured(code):
                        leading.append(idx)
                    else:
                        break

                if limit < len(leading):
                    leading = leading[: len(leading) - limit]
                else:
                    leading = []

                # C-terminus
                trailing = []
                for offset, code in enumerate(reversed(ss_string)):
                    if _is_unstructured(code):
                        trailing.append(len(ss_string) - 1 - offset)
                    else:
                        break
                trailing = list(reversed(trailing))
                if limit < len(trailing):
                    trailing = trailing[: len(trailing) - limit]
                else:
                    trailing = []

                remove_indexes = sorted(set(leading + trailing))

                if not remove_indexes:
                    continue

                # Only consider amino-acid polymer residues so ligands/nucleic acids survive trimming
                protein_residues = [
                    r
                    for r in chain_obj.get_residues()
                    if r.id[0] == " " and is_aa(r, standard=False)
                ]
                to_remove = [
                    protein_residues[i]
                    for i in remove_indexes
                    if i < len(protein_residues)
                ]

                removed_labels = []
                for residue in to_remove:
                    if verbose:
                        resseq = residue.id[1]
                        icode = residue.id[2].strip()
                        label = f"{model}:{chain_id}:{residue.get_resname().strip()} {resseq}{icode}"
                        removed_labels.append(label)
                    chain_obj.detach_child(residue.id)

                if verbose and removed_labels:
                    print(
                        "[removeTerminalUnstructuredRegions] Removed residues: "
                        + ", ".join(removed_labels)
                    )

        self.getModelsSequences()
        self.calculateSecondaryStructure()

    def removeTerminiByConfidenceScore(
        self,
        confidence_threshold=70.0,
        keep_up_to=5,
        lr=None,
        ur=None,
        renumber=False,
        verbose=True,
        output=None):
        """
        Remove terminal regions with low confidence scores and optionally trim residues by range.

        Parameters:
            confidence_threshold : float
                AlphaFold confidence threshold to consider residues as having a low score.
            keep_up_to : int
                If any terminal region is no larger than this value it will be kept.
            lr : dict, optional
                Dictionary specifying the lower range of residue indices to keep per model.
            ur : dict, optional
                Dictionary specifying the upper range of residue indices to keep per model.
            renumber : bool
                Whether to renumber residues after trimming.
            verbose : bool
                Whether to print warnings and updates.
            output : str, optional
                File path to save the modified structure.
        """
        remove_models = set()
        for model in self.models_names:
            atoms = [a for a in self.structures[model].get_atoms()]
            bfactors = [a.bfactor for a in atoms]

            if np.average(bfactors) == 0:
                if verbose:
                    print(
                        f"Warning: model {model} has no atom with the selected confidence!"
                    )
                remove_models.add(model)
                continue

            something = False
            n_terminus = {}
            c_terminus = {}

            for chain in self.structures[model].get_chains():
                chain_id = chain.get_id()
                polymer_atoms = [
                    atom
                    for atom in chain.get_atoms()
                    if atom.get_parent().id[0] == " "
                ]

                if not polymer_atoms:
                    continue

                chain_n = []
                for atom in polymer_atoms:
                    if atom.bfactor < confidence_threshold:
                        residue_id = atom.get_parent().id
                        if residue_id not in chain_n:
                            chain_n.append(residue_id)
                    else:
                        something = True
                        break

                chain_c = []
                for atom in reversed(polymer_atoms):
                    if atom.bfactor < confidence_threshold:
                        residue_id = atom.get_parent().id
                        if residue_id not in chain_c:
                            chain_c.append(residue_id)
                    else:
                        something = True
                        break

                if keep_up_to is not None and len(chain_n) <= keep_up_to:
                    chain_n = []
                if keep_up_to is not None and len(chain_c) <= keep_up_to:
                    chain_c = []

                if chain_n:
                    n_terminus[chain_id] = set(chain_n)
                if chain_c:
                    c_terminus[chain_id] = set(chain_c)

            if not something:
                if verbose and model not in remove_models:
                    print(
                        f"Warning: model {model} has no atom with the selected confidence!"
                    )
                remove_models.add(model)
                continue

            model_lr = lr.get(model, None) if lr else None
            model_ur = ur.get(model, None) if ur else None

            for c in self.structures[model].get_chains():
                chain_id = c.get_id()
                remove_this = []
                chain_n_ids = n_terminus.get(chain_id, set())
                chain_c_ids = c_terminus.get(chain_id, set())

                for r in c.get_residues():
                    if (
                        r.id in chain_n_ids
                        or r.id in chain_c_ids
                        or (
                            model_lr is not None
                            and model_ur is not None
                            and r.id[1] not in range(model_lr, model_ur + 1)
                        )
                    ):
                        remove_this.append(r)
                chain = c
                for r in remove_this:
                    chain.detach_child(r.id)

            if renumber:
                for c in self.structures[model].get_chains():
                    for i, r in enumerate(c):
                        r.id = (r.id[0], i + 1, r.id[2])

        for model in remove_models:
            self.removeModel(model)

        if output:
            for model in self.models_names:
                io.set_structure(self.structures[model])
                io.save(output)

        self.getModelsSequences()

        # self.calculateSecondaryStructure(_save_structure=True)

        # Missing save models and reload them to take effect.

    def getFoldseekMappingToReference(
        self,
        reference_pdb: str,
        min_prob: float = 0.0,
        threads: int | None = None,
        foldseek_exe: str = "foldseek",
        verbose: bool = False,
        tmp_dir: str | None = None,
    ) -> dict[str, dict[int, int]]:
        """
        Run Foldseek once (batch) using the *current in-memory* models.

        Steps:
          1) Save current structures to a temporary directory (fresh PDBs).
          2) Run: foldseek easy-search <ref> <temp_dir_of_targets>
          3) Parse best hit per target and build a mapping:
               { target_local_poly_idx (0-based) -> ref_local_poly_idx (0-based) }
          4) Return a dict: { <target_basename>: mapping }. Also tries to set
               protein.foldseek_mapping for convenience (best-effort).

        Parameters
        ----------
        tmp_dir : str | None
            Optional directory to use instead of a temporary one. If provided, results
            (including intermediate files) are left on disk for inspection.
        """
        import os
        import shutil
        import subprocess
        import tempfile

        if shutil.which(foldseek_exe) is None:
            raise FileNotFoundError(
                "Foldseek not found in PATH. Install it or pass 'foldseek_exe'."
            )

        tmpdir_cm = None
        if tmp_dir is None:
            tmpdir_cm = tempfile.TemporaryDirectory()
            tmpdir = tmpdir_cm.name
        else:
            tmpdir = tmp_dir
            os.makedirs(tmpdir, exist_ok=True)

        try:
            targets_dir = os.path.join(tmpdir, "targets")
            if os.path.exists(targets_dir):
                shutil.rmtree(targets_dir)
            os.makedirs(targets_dir, exist_ok=True)

            # Save *current* in-memory structures
            self.saveModels(targets_dir)

            target_files = [
                os.path.join(targets_dir, f)
                for f in os.listdir(targets_dir)
                if f.lower().endswith((".pdb", ".cif", ".mmcif"))
            ]
            if not target_files:
                raise RuntimeError(
                    "No target structures were saved to the temporary directory."
                )

            basename_to_obj = {}
            name_index = {}
            for p in getattr(self, "proteins", []):
                candidates = []
                for attr in ("id", "name", "model_id", "model", "label"):
                    val = getattr(p, attr, None)
                    if val:
                        candidates.append(str(val))
                for c in candidates:
                    name_index[c.lower()] = p

            for tf in target_files:
                bn = os.path.basename(tf)
                stem = os.path.splitext(bn)[0].lower()
                if stem in name_index:
                    basename_to_obj[bn] = name_index[stem]

            out_tsv = os.path.join(tmpdir, "foldseek.tsv")

            # Include qlen/tlen so we know full ref length
            fmt = "query,target,prob,qstart,qend,tstart,tend,alnlen,qaln,taln,qlen,tlen"
            cmd = [
                foldseek_exe,
                "easy-search",
                reference_pdb,
                targets_dir,
                out_tsv,
                tmpdir,
                "--alignment-type",
                "1",
                "--format-output",
                fmt,
                "--exhaustive-search",
                "1",
            ]
            if threads and threads > 0:
                cmd += ["--threads", str(threads)]

            if verbose:
                print("[foldseek] running:", " ".join(cmd))
                print(f"[foldseek] TSV: {out_tsv}")
            try:
                cp = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if verbose and cp.stderr.strip():
                    tail = "\n".join(cp.stderr.strip().splitlines()[-8:])
                    if tail:
                        print("[foldseek] stderr (tail):\n" + tail)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Foldseek failed with code {e.returncode}.\nSTDERR:\n{e.stderr}"
                ) from e

            if not os.path.exists(out_tsv) or os.path.getsize(out_tsv) == 0:
                raise RuntimeError("Foldseek produced no results (empty TSV).")

            best_by_target: dict[str, dict] = {}
            with open(out_tsv, "r") as fh:
                for raw in fh:
                    if verbose:
                        print(raw, end="")
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    # Expecting 12 fields now
                    if len(parts) < 12:
                        continue
                    rec = {
                        "query":   parts[0],
                        "target":  os.path.basename(parts[1]),
                        "prob":    float(parts[2]),
                        "qstart":  int(parts[3]),
                        "qend":    int(parts[4]),
                        "tstart":  int(parts[5]),
                        "tend":    int(parts[6]),
                        "alnlen":  int(parts[7]),
                        "qaln":    parts[8],
                        "taln":    parts[9],
                        "qlen":    int(parts[10]),
                        "tlen":    int(parts[11]),
                    }
                    cur = best_by_target.get(rec["target"])
                    if (cur is None) or (rec["prob"] > cur["prob"]):
                        best_by_target[rec["target"]] = rec

            results: dict[str, dict[int, int]] = {}
            for tgt_bn, rec in best_by_target.items():
                prob = rec["prob"]
                if prob < min_prob:
                    if verbose:
                        print(f"[foldseek] {tgt_bn}: prob={prob:.2f} < {min_prob} → skipped")
                    continue

                qaln, taln = rec["qaln"], rec["taln"]
                if len(qaln) != len(taln):
                    if verbose:
                        print(
                            f"[foldseek] {tgt_bn}: alignment length mismatch (qaln/taln) → skipped"
                        )
                    continue

                # Build raw mapping from alignment strings (0-based local indices)
                qpos = rec["qstart"] - 1
                tpos = rec["tstart"] - 1
                mapping: dict[int, int] = {}
                for qa, ta in zip(qaln, taln):
                    q_idx = None
                    t_idx = None
                    if qa != "-":
                        q_idx = qpos
                        qpos += 1
                    if ta != "-":
                        t_idx = tpos
                        tpos += 1
                    if (q_idx is not None) and (t_idx is not None):
                        if t_idx not in mapping:
                            mapping[t_idx] = q_idx

                # Keep the raw mapping from Foldseek (no clipping)
                results[tgt_bn] = mapping

                if verbose:
                    print(f"[foldseek] {tgt_bn}: prob={prob:.2f}  mapped={len(mapping)}")

                p = basename_to_obj.get(tgt_bn)
                if p is not None:
                    try:
                        setattr(p, "foldseek_mapping", mapping)
                    except Exception:
                        pass

            if verbose:
                print(
                    f"[foldseek] completed. {len(results)} mappings above prob ≥ {min_prob}."
                )

            return results
        finally:
            if tmpdir_cm is not None:
                tmpdir_cm.cleanup()


    def removeNotAlignedRegions(
        self,
        ref_structure,
        max_ca_ca=5.0,
        remove_low_confidence_unaligned_loops=False,
        confidence_threshold=50.0,
        min_loop_length=10,
        keep_aligned_fold_until_low_confidence=False,
        **kwargs,
    ):
        """
        Default behavior: define the fold core using the target residues aligned to the reference termini (first and last aligned reference positions per chain) and remove only terminal regions outside that span (no internal removals).
        Special modes:
        - keep_aligned_fold_until_low_confidence=True (existing)
        - remove_low_confidence_unaligned_loops=True (existing)

        Remove regions not structurally aligned to a reference, or (optional) keep only the
        aligned fold extended through high-confidence residues up to the first low-confidence
        region (per chain).

        Modes
        -----
        - Default:
            For each chain, identify the target residues aligned to the smallest and largest
            reference residue indices, keep the contiguous span between them, and trim only
            termini outside this anchor-defined core. Internal unaligned residues inside the
            span are preserved.
        - Fold-keep mode (keep_aligned_fold_until_low_confidence=True):
            For each chain, identify the aligned core (positions with structural alignment),
            then extend the kept region on the N and C sides across any contiguous residues
            whose mean per-residue confidence (AlphaFold pLDDT stored in B-factor) is
            >= confidence_threshold. Stop extending at the first low-confidence residue in
            each direction. Delete residues outside the kept region.
        - Loop-prune mode (remove_low_confidence_unaligned_loops=True):
            Retain aligned residues plus termini while pruning internal unaligned loops that are both
            sufficiently long and below the confidence_threshold.

        Parameters
        ==========
        ref_structure : str or Bio.PDB.Structure.Structure
            Path to the input PDB or model name to use as reference. Otherwise a Bio.PDB.Structure object
            can be given.
        max_ca_ca : float
            Maximum CA-CA distance for two residues to be considered aligned.
        remove_low_confidence_unaligned_loops : bool
            Remove not aligned loops with low confidence from the model
        confidence_threshold : float
            Threshold to consider a loop region not algined as low confidence for their removal.
        min_loop_length : int
            Length of the internal unaligned region to be considered a loop.
        keep_aligned_fold_until_low_confidence : bool
            When True, keep aligned cores and extend across high-confidence residues, removing
            all other polymeric residues per chain.

        Returns
        =======
        None

        Side effects:
            - Mutates structures in-place by deleting residues.
            - Calls self.getModelsSequences() at the end.
        """
        debug: bool = kwargs.pop("debug", False)
        dry_run: bool = kwargs.pop("dry_run", False)  # if True, compute but do not modify models
        tmp_dir: str | None = kwargs.pop("tmp_dir", None)

        if kwargs:
            warnings.warn(
                f"removeNotAlignedRegions: ignoring unexpected kwargs {list(kwargs.keys())}"
            )

        reference_input = ref_structure

        if isinstance(ref_structure, str):
            if ref_structure.endswith('.pdb'):
                ref_structure = prepare_proteins._readPDB('ref', ref_structure)
            else:
                if ref_structure in self.models_names:
                    ref_structure = self.structures[ref_structure]
                else:
                    raise ValueError('Reference structure was not found in models')
        elif not isinstance(ref_structure, PDB.Structure.Structure):
            raise ValueError(
                'ref_structure should be a  Bio.PDB.Structure.Structure or string object'
            )

        reference_label = (
            reference_input
            if isinstance(reference_input, str)
            else getattr(reference_input, "id", repr(reference_input))
        )

        _dbg(debug, "\n[removeNotAlignedRegions] reference_pdb: {}", reference_label)
        _dbg(debug, "[options] max_ca_ca: {}", max_ca_ca)
        _dbg(debug, "[options] confidence_threshold: {}", confidence_threshold)
        _dbg(debug, "[options] min_loop_length: {}", min_loop_length)
        _dbg(debug, "[options] remove_low_confidence_unaligned_loops: {}", remove_low_confidence_unaligned_loops)
        _dbg(debug, "[options] keep_aligned_fold_until_low_confidence: {}", keep_aligned_fold_until_low_confidence)
        _dbg(debug, "[options] dry_run: {}", dry_run)
        if tmp_dir:
            _dbg(debug, "[options] foldseek tmp_dir: {}", tmp_dir)

        # --- Conservative terminal-seed clipping thresholds (local to this function) ---
        EDGE_SEED_MAX = 12        # max residues in a terminal "seed" to consider clipping
        EDGE_GAP_MIN = 20         # min gap (in target indices) separating seed from main cluster
        MAIN_MIN = 120            # main cluster must have at least this many aligned residues
        REF_EDGE_WINDOW = 25      # first/last N reference residues = "edges"
        REF_EDGE_MIN_HITS = 8     # need >= this many hits in a ref edge to keep that seed
        REQUIRE_REF_EDGE_SUPPORT = True  # if True, weak ref-edge support allows clipping
        # ------------------------------------------------------------------------------

        ref_ca_atoms = [a for a in ref_structure.get_atoms() if a.name == "CA"]
        ref_ca_coord = np.array([a.coord for a in ref_ca_atoms])
        _dbg(debug, "[reference] CA atoms: {}", len(ref_ca_atoms))

        # Count reference polymer residues (for ref-edge support checks)
        ref_poly_len = sum(1 for r in ref_structure.get_residues() if r.id[0] == " ")

        import tempfile
        from Bio.PDB import PDBIO

        reference_path_for_foldseek = None
        temp_reference_path = None
        if isinstance(reference_input, str) and os.path.exists(reference_input):
            reference_path_for_foldseek = reference_input
        else:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
            os.close(tmp_fd)
            io = PDBIO()
            io.set_structure(ref_structure)
            io.save(tmp_path)
            reference_path_for_foldseek = tmp_path
            temp_reference_path = tmp_path

        try:
            foldseek_mappings = self.getFoldseekMappingToReference(
                reference_path_for_foldseek,
                min_prob=0.0,
                threads=None,
                foldseek_exe="foldseek",
                verbose=debug,
                tmp_dir=tmp_dir,
            )
        finally:
            if temp_reference_path and os.path.exists(temp_reference_path):
                try:
                    os.remove(temp_reference_path)
                except OSError:
                    pass

        foldseek_mappings_by_stem = {
            Path(k).stem: v for k, v in foldseek_mappings.items()
        }

        def _mean_residue_bfactor(residue):
            vals = [a.bfactor for a in residue.get_atoms()]
            return float(np.mean(vals)) if vals else 0.0

        for model in self:
            structure = self.structures[model]

            polymer_positions = []
            for chain in structure.get_chains():
                for res in chain.get_residues():
                    if res.id[0] == " ":
                        polymer_positions.append((chain.id, res))

            mapping = {}
            for ext in ("pdb", "cif", "mmcif"):
                key = f"{model}.{ext}"
                if key in foldseek_mappings:
                    mapping = dict(foldseek_mappings[key])
                    break
            if not mapping:
                mapping = dict(foldseek_mappings_by_stem.get(model, {}))
            if debug and not mapping:
                _dbg(debug, "[foldseek] model {}: no mapping retrieved; treating as unaligned", model)

            aligned_mask = np.zeros(len(polymer_positions), dtype=bool)
            out_of_range = []
            for tgt_idx in mapping.keys():
                if 0 <= tgt_idx < len(aligned_mask):
                    aligned_mask[tgt_idx] = True
                else:
                    out_of_range.append(tgt_idx)
            if out_of_range and debug:
                _dbg(
                    debug,
                    "[mapping] model {}: target indices out of range -> {} (poly_len={})",
                    model,
                    out_of_range,
                    len(aligned_mask),
                )

            if debug:
                try:
                    target_ca_coord = np.array(
                        [a.coord for a in structure.get_atoms() if a.name == "CA"]
                    )
                    _analyze_mapping_pairs(
                        mapping,
                        ref_ca_coord,
                        target_ca_coord,
                        max_ca_ca,
                        verbose_limit=30,
                    )
                except Exception as exc:
                    _dbg(
                        debug,
                        "[mapdiag] analysis failed for model {}: {}",
                        model,
                        exc,
                    )

            aligned_bool = aligned_mask.tolist()

            per_chain_indices = {}
            for idx, (cid, _) in enumerate(polymer_positions):
                per_chain_indices.setdefault(cid, []).append(idx)

            per_chain_res_index_bounds = {
                cid: (indices[0], indices[-1] + 1)
                for cid, indices in per_chain_indices.items()
                if indices
            }

            def _clip_terminal_seed_hits_for_chain(chain_pairs_sorted_by_tgt, ref_len, label=""):
                """
                chain_pairs_sorted_by_tgt: list of (ref_idx, tgt_idx) *for this chain*, sorted by tgt_idx.
                Returns (core_start, core_end, decision_info, dropped_t_indices_set).
                If no clipping is needed, returns the span of all hits and empty dropped set.
                """
                if not chain_pairs_sorted_by_tgt:
                    return None, None, {"reason": "no_hits", "label": label}, set()

                tgt_hits = [t for (r, t) in chain_pairs_sorted_by_tgt]
                ref_hits = [r for (r, t) in chain_pairs_sorted_by_tgt]

                clusters = []
                cs = ce = tgt_hits[0]
                cnt = 1
                for a, b in zip(tgt_hits[1:], tgt_hits[:-1]):
                    if a - b > EDGE_GAP_MIN:
                        clusters.append((cs, ce, cnt))
                        cs = ce = a
                        cnt = 1
                    else:
                        ce = a
                        cnt += 1
                clusters.append((cs, ce, cnt))

                span_start, span_end = tgt_hits[0], tgt_hits[-1]
                info = {
                    "initial_span": (span_start, span_end),
                    "clusters": clusters,
                    "label": label,
                }

                if len(clusters) == 1:
                    info["decision"] = "keep_span_single_cluster"
                    return span_start, span_end, info, set()

                main_idx = max(range(len(clusters)), key=lambda i: clusters[i][2])
                main_cs, main_ce, main_cnt = clusters[main_idx]
                info["main_cluster"] = (main_cs, main_ce, main_cnt)

                if main_cnt < MAIN_MIN:
                    info["decision"] = "keep_span_main_too_small"
                    return span_start, span_end, info, set()

                def seed_ref_hits_in_window(t_lo, t_hi, ref_lo, ref_hi):
                    c = 0
                    for (r, t) in chain_pairs_sorted_by_tgt:
                        if t_lo <= t <= t_hi and ref_lo <= r <= ref_hi:
                            c += 1
                    return c

                dropped = set()
                decision = "keep_span"
                core_start, core_end = span_start, span_end

                if main_idx > 0:
                    n_cs, n_ce, n_cnt = clusters[0]
                    gap_to_main = max(0, main_cs - n_ce)
                    ref_hits_N = seed_ref_hits_in_window(n_cs, n_ce, 0, max(0, REF_EDGE_WINDOW - 1))
                    ok_ref_seed = (
                        ref_hits_N < REF_EDGE_MIN_HITS
                    ) if REQUIRE_REF_EDGE_SUPPORT else True
                    if n_cnt <= EDGE_SEED_MAX and gap_to_main >= EDGE_GAP_MIN and ok_ref_seed:
                        for (r, t) in chain_pairs_sorted_by_tgt:
                            if n_cs <= t <= n_ce:
                                dropped.add(t)
                        core_start = main_cs
                        decision = f"clip_N_seed size={n_cnt} gap={gap_to_main} ref_edge_hits={ref_hits_N}"

                if main_idx < len(clusters) - 1:
                    c_cs, c_ce, c_cnt = clusters[-1]
                    gap_to_main = max(0, c_cs - main_ce)
                    ref_hits_C = seed_ref_hits_in_window(
                        c_cs, c_ce, max(0, ref_len - REF_EDGE_WINDOW), ref_len - 1
                    )
                    ok_ref_seed = (
                        ref_hits_C < REF_EDGE_MIN_HITS
                    ) if REQUIRE_REF_EDGE_SUPPORT else True
                    if c_cnt <= EDGE_SEED_MAX and gap_to_main >= EDGE_GAP_MIN and ok_ref_seed:
                        for (r, t) in chain_pairs_sorted_by_tgt:
                            if c_cs <= t <= c_ce:
                                dropped.add(t)
                        core_end = main_ce
                        if decision == "keep_span":
                            decision = f"clip_C_seed size={c_cnt} gap={gap_to_main} ref_edge_hits={ref_hits_C}"
                        else:
                            decision += f" + clip_C_seed size={c_cnt} gap={gap_to_main} ref_edge_hits={ref_hits_C}"

                info["decision"] = decision
                return core_start, core_end, info, dropped

            mapping_pairs = sorted(
                [(ref_idx, target_idx) for target_idx, ref_idx in mapping.items()],
                key=lambda pair: pair[1],
            )

            chains = [c.id for c in structure.get_chains()]
            total_residues = sum(1 for _ in structure.get_residues())
            _dbg(debug, "\n[model {}] chains: {}", model, chains)
            _dbg(debug, "[model {}] total residues: {}", model, total_residues)

            aligned_count = int(aligned_mask.sum())
            _dbg(debug, "[model {}] aligned residues: {}", model, aligned_count)

            try:
                per_chain_summary = []
                for ch_id, (start_idx, end_idx) in per_chain_res_index_bounds.items():
                    mask_slice = aligned_mask[start_idx:end_idx]
                    size = end_idx - start_idx
                    per_chain_summary.append((ch_id, int(mask_slice.sum()), size))
                for ch_id, aln, size in per_chain_summary:
                    _dbg(
                        debug,
                        "[model {}] chain {}: aligned {}/{} ({:.1f}%)",
                        model,
                        ch_id,
                        aln,
                        size,
                        100 * aln / max(1, size),
                    )
            except Exception:
                pass

            try:
                best_len = 0
                best_start = -1
                cur = 0
                cur_start = 0
                for i, v in enumerate(aligned_mask):
                    if v:
                        if cur == 0:
                            cur_start = i
                        cur += 1
                        if cur > best_len:
                            best_len = cur
                            best_start = cur_start
                    else:
                        cur = 0
                if best_len > 0:
                    _dbg(
                        debug,
                        "[model {}] longest aligned segment: {} residues (idx {}..{})",
                        model,
                        best_len,
                        best_start,
                        best_start + best_len - 1,
                    )
            except Exception:
                pass

            try:
                mapping_len = len(mapping_pairs)
                _dbg(debug, "[model {}] mapping size: {}", model, mapping_len)
                if mapping_len <= 50:
                    _dbg(
                        debug,
                        "[model {}] mapping (ref_idx -> model_idx): {}",
                        model,
                        mapping_pairs,
                    )
            except Exception:
                pass

            mask_list = aligned_mask.tolist() if hasattr(aligned_mask, "tolist") else list(aligned_mask)
            runs = _mask_runs(mask_list)

            # Compute residue deletions per chain
            indices_to_remove_by_chain = {cid: set() for cid in per_chain_indices}
            default_mode = False
            default_run_spans = []

            if keep_aligned_fold_until_low_confidence:
                for cid, idx_list in per_chain_indices.items():
                    if not idx_list:
                        continue

                    aligned_idxs = [k for k in idx_list if aligned_bool[k]]

                    # If no aligned residues in this chain, drop all polymer residues in this chain
                    if not aligned_idxs:
                        for k in idx_list:
                            indices_to_remove_by_chain[cid].add(k)
                        continue

                    # Core bounds (aligned region)
                    core_start = min(aligned_idxs)
                    core_end = max(aligned_idxs)

                    # Expand to N side through contiguous high-confidence residues
                    n_keep = core_start
                    cursor = core_start - 1
                    while cursor >= idx_list[0]:
                        _, res = polymer_positions[cursor]
                        if _mean_residue_bfactor(res) >= float(confidence_threshold):
                            n_keep = cursor
                            cursor -= 1
                        else:
                            break  # stop at first low-confidence residue

                    # Expand to C side through contiguous high-confidence residues
                    c_keep = core_end
                    cursor = core_end + 1
                    while cursor <= idx_list[-1]:
                        _, res = polymer_positions[cursor]
                        if _mean_residue_bfactor(res) >= float(confidence_threshold):
                            c_keep = cursor
                            cursor += 1
                        else:
                            break  # stop at first low-confidence residue

                    # Everything outside [n_keep, c_keep] (within this chain) is removed
                    keep_range = set(range(n_keep, c_keep + 1))
                    for k in idx_list:
                        if k not in keep_range:
                            indices_to_remove_by_chain[cid].add(k)

            elif remove_low_confidence_unaligned_loops:
                for cid, idx_list in per_chain_indices.items():
                    if not idx_list:
                        continue

                    # N-terminus run of unaligned
                    n_cut = []
                    for k in idx_list:
                        if not aligned_bool[k]:
                            n_cut.append(k)
                        else:
                            break

                    # C-terminus run of unaligned
                    c_cut = []
                    for k in reversed(idx_list):
                        if not aligned_bool[k]:
                            c_cut.append(k)
                        else:
                            break
                    c_cut.reverse()

                    idxs_to_remove = set(n_cut + c_cut)

                    # Identify internal region
                    start = 0
                    while start < len(idx_list) and not aligned_bool[idx_list[start]]:
                        start += 1
                    end = len(idx_list) - 1
                    while end >= 0 and not aligned_bool[idx_list[end]]:
                        end -= 1
                    if start < end:
                        run = []
                        internal_runs = []
                        for pos in idx_list[start : end + 1]:
                            if not aligned_bool[pos]:
                                run.append(pos)
                            else:
                                if run:
                                    internal_runs.append(run)
                                    run = []
                        if run:
                            internal_runs.append(run)

                        for run_idx_list in internal_runs:
                            if len(run_idx_list) < int(min_loop_length):
                                continue
                            residues_in_run = [polymer_positions[k][1] for k in run_idx_list]
                            run_conf = float(
                                np.mean([_mean_residue_bfactor(r) for r in residues_in_run])
                            )
                            if run_conf < float(confidence_threshold):
                                idxs_to_remove.update(run_idx_list)

                    indices_to_remove_by_chain[cid].update(idxs_to_remove)

            else:
                # Default: define core by reference-termini anchors (no internal removals).
                # Pick, per chain, the target residues aligned to the smallest and largest
                # reference indices; keep the contiguous span between them. Trim only termini.
                default_mode = True
                default_run_spans = []  # (start,end) terminal spans to drop across all chains

                for cid, idx_list in per_chain_indices.items():
                    if not idx_list:
                        continue

                    core = _term_anchored_core(mapping_pairs, idx_list)
                    if core is None:
                        # Nothing aligned to this chain → drop full chain slice
                        for k in idx_list:
                            indices_to_remove_by_chain[cid].add(k)
                        default_run_spans.append((idx_list[0], idx_list[-1]))
                        _dbg(debug, "[default-core] chain {}: no aligned anchors; dropping {}..{}", cid, idx_list[0], idx_list[-1])
                        continue

                    core_start, core_end = core

                    chain_set = set(idx_list)
                    try:
                        r_min, t_at_rmin = min(
                            ((r, t) for (r, t) in mapping_pairs if t in chain_set),
                            key=lambda x: x[0],
                        )
                        r_max, t_at_rmax = max(
                            ((r, t) for (r, t) in mapping_pairs if t in chain_set),
                            key=lambda x: x[0],
                        )
                        _dbg(
                            debug,
                            "[default-core] chain {}: anchors ref[min={}, max={}] → tgt[{}..{}] | core {}..{}",
                            cid,
                            r_min,
                            r_max,
                            t_at_rmin,
                            t_at_rmax,
                            core_start,
                            core_end,
                        )
                    except Exception:
                        pass

                    chain_pairs_tgt_sorted = sorted(
                        [(r, t) for (r, t) in mapping_pairs if t in chain_set],
                        key=lambda x: x[1],
                    )

                    if chain_pairs_tgt_sorted:
                        c_start2, c_end2, info2, dropped = _clip_terminal_seed_hits_for_chain(
                            chain_pairs_tgt_sorted, ref_poly_len, label=f"{model}:{cid}"
                        )
                        if (
                            c_start2 is not None
                            and c_end2 is not None
                            and info2.get("decision", "").startswith("clip_")
                        ):
                            _dbg(
                                debug,
                                "[foldseek-core] model {} chain {}: {} | anchor_core {}..{} -> pruned_core {}..{} (dropped {})",
                                model,
                                cid,
                                info2.get("decision"),
                                core_start,
                                core_end,
                                c_start2,
                                c_end2,
                                len(dropped),
                            )
                            core_start, core_end = c_start2, c_end2
                        elif c_start2 is not None and c_end2 is not None:
                            _dbg(
                                debug,
                                "[foldseek-core] model {} chain {}: {} | keeping anchor_core {}..{}",
                                model,
                                cid,
                                info2.get("decision"),
                                core_start,
                                core_end,
                            )

                    core_start = max(core_start, idx_list[0])
                    core_end = min(core_end, idx_list[-1])
                    if core_start > core_end:
                        core_start, core_end = core

                    if core_start > idx_list[0]:
                        for k in range(idx_list[0], core_start):
                            indices_to_remove_by_chain[cid].add(k)
                        default_run_spans.append((idx_list[0], core_start - 1))

                    if core_end < idx_list[-1]:
                        for k in range(core_end + 1, idx_list[-1] + 1):
                            indices_to_remove_by_chain[cid].add(k)
                        default_run_spans.append((core_end + 1, idx_list[-1]))

            unaligned_regions = []
            to_remove_by_chain = {}
            for cid, idxs in indices_to_remove_by_chain.items():
                if not idxs:
                    continue
                sorted_idxs = sorted(idxs)
                residues_for_chain = []
                start_idx = sorted_idxs[0]
                prev_idx = start_idx
                for idx in sorted_idxs[1:]:
                    if idx != prev_idx + 1:
                        start_res = polymer_positions[start_idx][1]
                        end_res = polymer_positions[prev_idx][1]
                        unaligned_regions.append(
                            (cid, start_res.id[1], end_res.id[1])
                        )
                        start_idx = idx
                    prev_idx = idx
                # flush final range
                start_res = polymer_positions[start_idx][1]
                end_res = polymer_positions[prev_idx][1]
                unaligned_regions.append((cid, start_res.id[1], end_res.id[1]))

                for idx in sorted_idxs:
                    residues_for_chain.append(polymer_positions[idx][1])
                to_remove_by_chain[cid] = residues_for_chain

            # Planned trimming report
            if default_mode:
                try:
                    default_drop_total = sum(
                        (end - start + 1) for start, end in default_run_spans
                    )
                    _dbg(
                        debug,
                        "[trim] mode='default' (keep contiguous aligned core; trim termini only) | drop_total={}",
                        default_drop_total,
                    )
                    _dbg(
                        debug,
                        "[trim] unaligned regions (count={}): {}",
                        len(default_run_spans),
                        default_run_spans,
                    )
                except Exception:
                    pass

            try:
                _dbg(
                    debug,
                    "[trim] unaligned regions by chain (count={}): {}",
                    len(unaligned_regions),
                    unaligned_regions,
                )
            except Exception:
                pass

            try:
                drop_by_chain = {}
                for (chain_id, start_idx, end_idx) in unaligned_regions:
                    drop_by_chain.setdefault(chain_id, 0)
                    drop_by_chain[chain_id] += (end_idx - start_idx + 1)
                for ch, n in drop_by_chain.items():
                    _dbg(debug, "[trim] chain {}: dropping {} residues", ch, n)
            except Exception:
                pass

            if dry_run:
                _dbg(debug, "[trim] dry_run=True — skipping structure mutation for model {}", model)
            else:
                # Apply deletions on the correct chain objects
                chains_map = {c.id: c for c in structure.get_chains()}
                for cid, residues in to_remove_by_chain.items():
                    chain_obj = chains_map.get(cid)
                    if not chain_obj or not residues:
                        continue
                    for res in residues:
                        chain_obj.detach_child(res.id)

                try:
                    new_total = sum(1 for _ in structure.get_residues())
                    _dbg(debug, "[post] model {} residues after trim: {}", model, new_total)
                except Exception:
                    pass

        # Refresh sequences once at the end
        self.getModelsSequences()

    def trimByRanges(self, ranges, renumber=False, verbose=True):
        """
        Trim models strictly by provided residue ID ranges.

        Parameters
        ==========
        ranges : dict
            Mapping of model name -> list of (start, end) tuples specifying
            inclusive residue ID ranges to KEEP. Residue IDs are the PDB
            numbering (r.id[1]) and are typically one-based. A single
            (start, end) tuple is also accepted instead of a list.
        renumber : bool
            If True, renumber residues in each chain sequentially starting at 1
            after trimming.
        verbose : bool
            If True, print basic warnings/info.
        """

        if not isinstance(ranges, dict):
            raise ValueError("ranges must be a dict: {model: [(start, end), ...]}")

        for model in list(self.models_names):
            if model not in ranges:
                if verbose:
                    print(f"trimByRanges: skipping model '{model}' (no ranges provided)")
                continue

            model_ranges = ranges[model]
            # Accept a single tuple as shorthand
            if isinstance(model_ranges, tuple) and len(model_ranges) == 2:
                model_ranges = [model_ranges]

            if not isinstance(model_ranges, (list, tuple)) or not all(
                isinstance(x, (list, tuple)) and len(x) == 2 for x in model_ranges
            ):
                raise ValueError(
                    f"ranges['{model}'] must be a list of (start, end) tuples or a single tuple"
                )

            # Normalize and sanity-check intervals (inclusive)
            keep_intervals = []
            for start, end in model_ranges:
                try:
                    s = int(start)
                    e = int(end)
                except Exception:
                    raise ValueError(
                        f"ranges['{model}'] contains non-integer bounds: {(start, end)}"
                    )
                if s > e:
                    s, e = e, s
                keep_intervals.append((s, e))

            # Perform trimming
            for c in self.structures[model].get_chains():
                to_remove = []
                for r in c.get_residues():
                    if r.id[0] != " ":
                        # Keep hetero residues (cofactors, ions, waters) untouched
                        continue
                    resid = r.id[1]
                    # Keep if resid within any interval
                    keep = False
                    for s, e in keep_intervals:
                        if s <= resid <= e:
                            keep = True
                            break
                    if not keep:
                        to_remove.append(r)

                for r in to_remove:
                    c.detach_child(r.id)

                if renumber:
                    new_resseq = 1
                    for r in c.get_residues():
                        if r.id[0] != " ":
                            continue
                        r.id = (r.id[0], new_resseq, r.id[2])
                        new_resseq += 1

        # Update sequences after modifications
        self.getModelsSequences()

    def alignModelsToReferencePDB(
        self,
        reference,
        output_folder,
        chain_indexes=None,
        trajectory_chain_indexes=None,
        reference_chain_indexes=None,
        alignment_mode="aligned",
        verbose=False,
        reference_residues=None,
    ):
        """
        Align all models to a reference PDB based on a sequence alignemnt.

        The chains are specified using their indexes. When the trajectories have
        corresponding chains use the option chain_indexes to specify the list of
        chains to align. Otherwise, specify the chains with trajectory_chain_indexes
        and reference_chain_indexes options. Note that the list of chain indexes
        must be corresponding.

        Parameters
        ==========
        reference : str
            Path to the reference PDB
        output_folder : str
            Path to the output folder to store models
        mode : str
            The mode defines how sequences are aligned. 'exact' for structurally
            aligning positions with exactly the same aminoacids after the sequence
            alignemnt or 'aligned' for structurally aligining sequences using all
            positions aligned in the sequence alignment.

        Returns
        =======
        rmsd : tuple
            A tuple containing the RMSD in Angstroms and the number of alpha-carbon
            atoms over which it was calculated.
        """

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        reference = md.load(reference)
        rmsd = {}

        # Check chain indexes input
        if isinstance(trajectory_chain_indexes, list):
            tci = {}
            for model in self.models_names:
                tci[model] = trajectory_chain_indexes
            trajectory_chain_indexes = tci

        for model in self.models_names:

            if verbose:
                print("Saving model: %s" % model)

            traj = md.load(self.models_paths[model])
            if trajectory_chain_indexes is None:
                rmsd[model] = MD.alignTrajectoryBySequenceAlignment(
                    traj,
                    reference,
                    chain_indexes=chain_indexes,
                    trajectory_chain_indexes=trajectory_chain_indexes,
                    reference_chain_indexes=reference_chain_indexes,
                    alignment_mode=alignment_mode,
                    reference_residues=reference_residues,
                )
            else:
                rmsd[model] = MD.alignTrajectoryBySequenceAlignment(
                    traj,
                    reference,
                    chain_indexes=chain_indexes,
                    trajectory_chain_indexes=trajectory_chain_indexes[model],
                    reference_chain_indexes=reference_chain_indexes,
                    alignment_mode=alignment_mode,
                    reference_residues=reference_residues,
                )

            # Get bfactors
            bfactors = np.array([a.bfactor for a in self.structures[model].get_atoms()])

            # Correct B-factors outside the -10 to 100 range accepted ny mdtraj
            bfactors = np.where(bfactors >= 100.0, 99.99, bfactors)
            bfactors = np.where(bfactors <= -10.0, -9.99, bfactors)

            traj.save(output_folder + "/" + model + ".pdb", bfactors=bfactors)

        return rmsd

    def positionLigandsAtCoordinate(
        self,
        coordinate,
        ligand_folder,
        output_folder,
        separator="-",
        overwrite=True,
        only_models=None,
        only_ligands=None,
    ):
        """
        Position a set of ligands into specific protein coordinates.

        Parameters
        ==========
        coordinate : tuple or dict
            New desired coordinates of the ligand
        ligand_folder : str
            Path to the ligands folder to store ligand molecules
        output_folder : str
            Path to the output folder to store models
        overwrite : bool
            Overwrite if structure file already exists.
        """

        # Create output directory
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if isinstance(only_models, str):
            only_models = [only_models]

        if isinstance(only_ligands, str):
            only_ligands = [only_ligands]

        # Copy script file to output directory
        _copyScriptFile(output_folder, "positionLigandAtCoordinate.py")

        for l in os.listdir(ligand_folder):
            if l.endswith(".mae"):
                ln = l.replace(".mae", "")
            elif l.endswith(".pdb"):
                ln = l.replace(".pdb", "")
            else:
                continue

            if not isinstance(only_ligands, type(None)):
                if ln not in only_ligands:
                    continue

            for model in self:

                if not isinstance(only_models, type(None)):
                    if model not in only_models:
                        continue

                self.docking_ligands.setdefault(model, [])
                self.docking_ligands[model].append(ln)

                if not os.path.exists(output_folder + "/" + model):
                    os.mkdir(output_folder + "/" + model)

                if (
                    os.path.exists(
                        output_folder
                        + "/"
                        + model
                        + "/"
                        + model
                        + separator
                        + ln
                        + separator
                        + "0.pdb"
                    )
                    and not overwrite
                ):
                    continue

                _saveStructureToPDB(
                    self.structures[model],
                    output_folder + "/" + model + "/" + model + separator + ln + ".pdb",
                )
                command = (
                    "run python3 " + output_folder + "/._positionLigandAtCoordinate.py "
                )
                command += (
                    output_folder + "/" + model + "/" + model + separator + ln + ".pdb "
                )
                command += ligand_folder + "/" + l + " "
                if isinstance(coordinate, dict):
                    coordinate_string = (
                        '"' + ",".join([str(x) for x in coordinate[model]]) + '"'
                    )
                elif isinstance(coordinate, tuple) and len(coordinate) == 3:
                    coordinate_string = (
                        '"' + ",".join([str(x) for x in coordinate]) + '"'
                    )
                else:
                    raise ValueError(
                        "coordinate needs to be a 3-element tuple of integers or dict."
                    )
                if "-" in coordinate_string:
                    coordinate_string = coordinate_string.replace("-", "\-")
                command += coordinate_string
                command += ' --separator "' + separator + '" '
                command += " --pele_poses\n"
                os.system(command)

    def createMetalConstraintFiles(
        self, job_folder, sugars=False, params_folder=None, models=None
    ):
        """
        Create metal constraint files.

        Parameters
        ==========
        job_folder : str
            Folder path where to place the constraint files.
        sugars : bool
            Use carbohydrate aware Rosetta PDB reading.
        params_folder : str
            Path to a folder containing a set of params file to be employed.
        models : list
            Only consider models inside the given list.
        """

        # Create mutation job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")

        output_folder = job_folder + "/cst_files"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Copy models to input folder
        self.saveModels(job_folder + "/input_models", models=models)

        # Copy embeddingToMembrane.py script
        _copyScriptFile(job_folder, "createMetalConstraints.py", subfolder="pyrosetta")

        jobs = []
        for model in self:

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            command = "cd " + output_folder + "\n"
            command += "python ../._createMetalConstraints.py "
            command += "../input_models/" + model + ".pdb "
            command += "metal_" + model + ".cst "
            if sugars:
                command += "--sugars "
            if params_folder != None:
                command += "--params_folder " + params_folder + " "
            command += "\ncd " + "../" * len(output_folder.split("/")) + "\n"
            jobs.append(command.rstrip("\n") + "\n")

        return jobs

    def createMutants(
        self,
        job_folder,
        mutants,
        variant_fastas=None,
        nstruct=100,
        relax_cycles=0,
        cst_optimization=True,
        executable="rosetta_scripts.mpi.linuxgccrelease",
        sugars=False,
        param_files=None,
        parallelisation="srun",
        mpi_command=None,
        cpus=None,
    ):
        """
        Create mutations from protein models. Mutations (mutants) must be given as a nested dictionary
        with each protein as the first key and the name of the particular mutant as the second key.
        The value of each inner dictionary is a list containing the mutations, with each mutation
        described by a 2-element tuple (residue_id, aa_to_mutate). E.g., (73, 'A').

        Parameters
        ==========
        job_folder : str
            Folder path where to place the mutation job.
        mutants : dict
            Mutants specification. Two accepted formats (can be mixed per model):
              1) Nested dict: {model: {mutant_name: [(resid, new_aa), ...]}, ...}
              2) FASTA dict: {model: "/path/to/variants.fasta", ...} where each
                 FASTA entry header becomes the mutant name. For multi-chain
                 models, sequences must be separated by '/' per chain in chain
                 order. Entries must match base length and only use standard AAs.
                 Name collisions error.
        variant_fastas : dict, optional
            Same as FASTA dict above; kept for backwards compatibility. Collides
            with a FASTA entry in `mutants` for the same model -> error.
        relax_cycles : int
            Apply this number of relax cycles (default:0, i.e., no relax).
        nstruct : int
            Number of structures to generate when relaxing mutant
        param_files : list
            Params file to use when reading model with Rosetta.
        sugars : bool
            Use carbohydrate aware Rosetta optimization
        parallelisation : str, optional
            How to launch Rosetta. "srun", "mpirun", or None for a single process.
            Defaults to "srun". If using "mpirun", `cpus` must be provided.
        mpi_command : str, optional
            Deprecated compatibility alias. Accepted values: "slurm", "openmpi", or
            None. When provided, it is translated to `parallelisation`.
        """
        if mpi_command not in (None, "slurm", "openmpi"):
            raise ValueError("mpi_command must be one of: None, 'slurm', 'openmpi'")

        if not isinstance(mutants, dict):
            raise ValueError("mutants must be a dictionary mapping model -> variants or model -> fasta")

        # Create mutation job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")
        if not os.path.exists(job_folder + "/flags"):
            os.mkdir(job_folder + "/flags")
        if not os.path.exists(job_folder + "/xml"):
            os.mkdir(job_folder + "/xml")
        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        if variant_fastas is not None and not isinstance(variant_fastas, dict):
            raise ValueError("variant_fastas must be a dictionary mapping model -> fasta path")

        # Split inputs into explicit mutant dicts and fasta sources
        explicit_mutants = {}
        fasta_sources = {}
        for model, spec in mutants.items():
            if isinstance(spec, str):
                fasta_sources[model] = spec
            elif isinstance(spec, dict):
                explicit_mutants[model] = spec
            else:
                raise ValueError(
                    f"Invalid mutants entry for model {model}: expected dict of variants or fasta path string"
                )

        if variant_fastas:
            for model, fasta_path in variant_fastas.items():
                if model in fasta_sources:
                    raise ValueError(
                        f"FASTA source for model {model} provided both in mutants and variant_fastas"
                    )
                fasta_sources[model] = fasta_path

        # Ensure sequences are available for comparison
        requested_models = set(explicit_mutants.keys()) | set(fasta_sources.keys())
        if (
            not getattr(self, "sequences", None)
            or not getattr(self, "chain_sequences", None)
            or any(m not in self.sequences for m in requested_models)
        ):
            self.getModelsSequences()

        # Augment mutants with variants defined in fasta files
        all_mutants = copy.deepcopy(explicit_mutants)
        if fasta_sources:
            standard_aas = set("ACDEFGHIKLMNPQRSTVWY")

            def _ensure_model_sequences(model):
                if model not in self.sequences:
                    raise ValueError(f"Model {model} is not loaded; cannot read sequences for FASTA variants.")
                # When sequences are stored per chain, drop chains with empty/non-protein sequences
                if isinstance(self.sequences[model], dict):
                    chain_ids = []
                    chain_seqs = []
                    for cid, seq in self.chain_sequences[model].items():
                        if seq:
                            chain_ids.append(cid)
                            chain_seqs.append(seq)
                    if not chain_ids:
                        raise ValueError(f"No protein chains with sequences found for model {model}")
                    # If only one valid chain remains, treat it as single-chain for FASTA parsing
                    if len(chain_ids) == 1:
                        return [None], [chain_seqs[0]]
                    return chain_ids, chain_seqs
                return [None], [self.sequences[model]]

            for model, fasta_path in fasta_sources.items():
                if not os.path.exists(fasta_path):
                    raise FileNotFoundError(f"FASTA file for model {model} not found: {fasta_path}")

                chain_ids, chain_sequences = _ensure_model_sequences(model)
                variant_entries = alignment.readFastaFile(fasta_path)
                if not variant_entries:
                    raise ValueError(f"No sequences found in FASTA file for model {model}: {fasta_path}")

                if model not in all_mutants:
                    all_mutants[model] = {}

                seen_names = set()
                for variant_name, raw_seq in variant_entries.items():
                    if variant_name in seen_names:
                        raise ValueError(f"Duplicate variant name '{variant_name}' in FASTA for model {model}")
                    if variant_name in all_mutants[model]:
                        raise ValueError(
                            f"Variant name collision for model {model}: '{variant_name}' already defined in mutants dict"
                        )
                    seen_names.add(variant_name)

                    seq = raw_seq.upper()
                    if len(chain_ids) == 1:
                        if "/" in seq:
                            raise ValueError(
                                f"Unexpected '/' chain separator in single-chain model {model} for variant {variant_name}"
                            )
                        if len(seq) != len(chain_sequences[0]):
                            raise ValueError(
                                f"Length mismatch for model {model} variant {variant_name}: "
                                f"{len(seq)} vs base {len(chain_sequences[0])}"
                            )
                        if any(res not in standard_aas for res in seq):
                            raise ValueError(
                                f"Non-standard amino acid found in variant {variant_name} for model {model}"
                            )
                        diffs = [
                            (idx + 1, v_res)
                            for idx, (ref_res, v_res) in enumerate(zip(chain_sequences[0], seq))
                            if ref_res != v_res
                        ]
                    else:
                        split_seq = seq.split("/")
                        if len(split_seq) != len(chain_ids):
                            raise ValueError(
                                f"Chain count mismatch for model {model} variant {variant_name}: "
                                f"found {len(split_seq)} chains, expected {len(chain_ids)}"
                            )
                        diffs = []
                        offset = 0
                        for chain_idx, (ref_chain_seq, var_chain_seq) in enumerate(zip(chain_sequences, split_seq)):
                            if len(var_chain_seq) != len(ref_chain_seq):
                                raise ValueError(
                                    f"Length mismatch in chain {chain_ids[chain_idx]} for model {model} variant {variant_name}: "
                                    f"{len(var_chain_seq)} vs base {len(ref_chain_seq)}"
                                )
                            if any(res not in standard_aas for res in var_chain_seq):
                                raise ValueError(
                                    f"Non-standard amino acid in chain {chain_ids[chain_idx]} for model {model} variant {variant_name}"
                                )
                            for pos, (ref_res, v_res) in enumerate(zip(ref_chain_seq, var_chain_seq)):
                                if ref_res != v_res:
                                    diffs.append((offset + pos + 1, v_res))
                            offset += len(ref_chain_seq)

                    all_mutants[model][variant_name] = diffs

        # Save considered models
        considered_models = list(all_mutants.keys())
        self.saveModels(job_folder + "/input_models", models=considered_models)

        # Prepare params files if provided (support folder of params)
        params_output_dir = None
        param_paths = []
        if param_files is not None:
            params_output_dir = Path(job_folder) / "params"
            params_output_dir.mkdir(exist_ok=True)

            if isinstance(param_files, (str, os.PathLike)) and Path(param_files).is_dir():
                param_dir = Path(param_files)
                param_paths = sorted(param_dir.glob("*.params"))
                if not param_paths:
                    raise ValueError(f"No .params files found in directory {param_files}")
                # Copy accompanying PDBs if present (e.g., conformers)
                for pdb_file in param_dir.glob("*.pdb"):
                    shutil.copyfile(pdb_file, params_output_dir / pdb_file.name)
            else:
                if isinstance(param_files, (str, os.PathLike)):
                    param_paths = [Path(param_files)]
                else:
                    param_paths = [Path(p) for p in param_files]

            for p in param_paths:
                if not p.exists():
                    raise FileNotFoundError(f"Params file not found: {p}")
                shutil.copyfile(p, params_output_dir / p.name)

        jobs = []

        # Create all-atom score function
        score_fxn_name = "ref2015"
        sfxn = rosettaScripts.scorefunctions.new_scorefunction(
            score_fxn_name, weights_file=score_fxn_name
        )

        # Resolve parallelisation / mpi_command compatibility
        user_disabled_parallelisation = parallelisation is None
        if not user_disabled_parallelisation:
            # Allow legacy mpi_command to override when provided
            if parallelisation == "srun" and mpi_command == "openmpi":
                parallelisation = "mpirun"
            elif parallelisation == "mpirun" and mpi_command == "slurm":
                parallelisation = "srun"

        if user_disabled_parallelisation:
            if cpus is not None:
                raise ValueError("cpus is only used when parallelisation='mpirun'")
            mpi_prefix = ""
        else:
            if parallelisation is None and mpi_command == "openmpi":
                parallelisation = "mpirun"
            elif parallelisation is None and mpi_command == "slurm":
                parallelisation = "srun"

            if parallelisation not in (None, "srun", "mpirun"):
                raise ValueError("parallelisation must be one of: None, 'srun', 'mpirun'")

            if parallelisation == "mpirun":
                if not isinstance(cpus, int):
                    raise ValueError("You must define the number of CPU when using mpirun")
                mpi_prefix = f"mpirun -np {cpus} "
            elif parallelisation == "srun":
                if cpus is not None:
                    raise ValueError("CPUs can only be set when using mpirun parallelisation")
                mpi_prefix = "srun "
            else:
                mpi_prefix = ""

        for model in self.models_names:

            # Skip models not in given mutants
            if model not in considered_models:
                continue

            if not os.path.exists(job_folder + "/output_models/" + model):
                os.mkdir(job_folder + "/output_models/" + model)

            # Iterate each mutant
            for mutant in all_mutants[model]:

                if not isinstance(all_mutants[model][mutant], list):
                    raise ValueError('Mutations for a particular variant should be given as a list of tuples!')

                # Create xml mutation (and minimization) protocol
                xml = rosettaScripts.xmlScript()
                protocol = []

                # Add score function
                xml.addScorefunction(sfxn)

                for m in all_mutants[model][mutant]:
                    mutate = rosettaScripts.movers.mutate(
                        name="mutate_" + str(m[0]),
                        target_residue=m[0],
                        new_residue=_one_to_three(m[1]),
                    )
                    xml.addMover(mutate)
                    protocol.append(mutate)

                if relax_cycles:
                    # Create fastrelax mover
                    relax = rosettaScripts.movers.fastRelax(
                        repeats=relax_cycles, scorefxn=sfxn
                    )
                    xml.addMover(relax)
                    protocol.append(relax)
                else:
                    # Turn off more than one structure when relax is not performed
                    nstruct = 1

                # Set protocol
                xml.setProtocol(protocol)

                # Add scorefunction output
                xml.addOutputScorefunction(sfxn)

                # Write XMl protocol file
                xml.write_xml(job_folder + "/xml/" + model + "_" + mutant + ".xml")

                # Create options for minimization protocol
                flags = rosettaScripts.flags(
                    "../../xml/" + model + "_" + mutant + ".xml",
                    nstruct=nstruct,
                    s="../../input_models/" + model + ".pdb",
                    output_silent_file=model + "_" + mutant + ".out",
                )

                # Add relaxation with constraints options and write flags file
                if cst_optimization and relax_cycles:
                    flags.add_relax_cst_options()
                else:
                    flags.add_relax_options()

                # Add path to params files
                if param_paths:
                    flags.addOption("in:file:extra_res_path", "../../params")

                if sugars:
                    flags.addOption("include_sugars")
                    flags.addOption("alternate_3_letter_codes", "pdb_sugar")
                    flags.addOption("write_glycan_pdb_codes")
                    flags.addOption("auto_detect_glycan_connections")
                    flags.addOption("maintain_links")

                flags.write_flags(
                    job_folder + "/flags/" + model + "_" + mutant + ".flags"
                )

                command = "cd " + job_folder + "/output_models/" + model + "\n"
                command += (
                    mpi_prefix
                    + executable
                    + " @ "
                    + "../../flags/"
                    + model
                    + "_"
                    + mutant
                    + ".flags\n"
                )
                command += "cd ../../..\n"
                jobs.append(command.rstrip("\n") + "\n")

        return jobs

    def createDisulfureBond(
        self,
        job_folder,
        cys_dic,
        nstruct=100,
        relax_cycles=0,
        cst_optimization=True,
        executable="rosetta_scripts.mpi.linuxgccrelease",
        param_files=None,
        mpi_command="slurm",
        cpus=None,
        remove_existing=False,
        repack=True,
        scorefxn="ref2015",
    ):
        """
        Create Disulfure bonds from protein models. Cysteine residues must be given as a nested dictionary
        with each protein as the first key and the name of the particular mutant as the second key.
        The value of each inner dictionary is a list containing only one element string with the cisteine pairs with : separator.
        It is recommended to use absolute pose positions.

        The mover is ForceDisulfides

        Parameters
        ==========
        job_folder : str
            Folder path where to place the mutation job.
        cys_dic : dict
            Dictionary specify the cysteine bonds to generate. # It is better to use the pose numbering to avoid
            problems with the chains. !!!!! IMPORTANT !!!!!! Adjust first the positions in a previous function
            as in the case of mutate residue.

        relax_cycles : int
            Apply this number of relax cycles (default:0, i.e., no relax).
        nstruct : int
            Number of structures to generate when relaxing mutant
        param_files : list
            Params file to use when reading model with Rosetta.
        """

        # Check both residues are cysteine

        mpi_commands = ["slurm", "openmpi", None]
        if mpi_command not in mpi_commands:
            raise ValueError(
                "Wrong mpi_command it should either: " + " ".join(mpi_commands)
            )

        if mpi_command == "openmpi" and not isinstance(cpus, int):
            raise ValueError("")

        # Create mutation job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")
        if not os.path.exists(job_folder + "/flags"):
            os.mkdir(job_folder + "/flags")
        if not os.path.exists(job_folder + "/xml"):
            os.mkdir(job_folder + "/xml")
        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Save considered models
        considered_models = list(cys_dic.keys())
        self.saveModels(job_folder + "/input_models", models=considered_models)

        jobs = []

        # Create all-atom score function
        score_fxn_name = "ref2015"
        sfxn = rosettaScripts.scorefunctions.new_scorefunction(
            score_fxn_name, weights_file=score_fxn_name
        )

        # Create and append execution command
        if mpi_command == None:
            mpi_command = ""
        elif mpi_command == "slurm":
            mpi_command = "srun "
        elif mpi_command == "openmpi":
            mpi_command = "mpirun -np " + str(cpus) + " "
        else:
            mpi_command = mpi_command + " "

        for model in self.models_names:

            # Skip models not in given cys_dic
            if model not in considered_models:
                continue

            if not os.path.exists(job_folder + "/output_models/" + model):
                os.mkdir(job_folder + "/output_models/" + model)

            # Iterate each mutant
            for mutant in cys_dic[model]:

                # Create xml mutation (and minimization) protocol
                xml = rosettaScripts.xmlScript()
                protocol = []

                # Add score function
                xml.addScorefunction(sfxn)

                for m in cys_dic[model][mutant]:
                    disulfide = rosettaScripts.movers.ForceDisulfides(
                        name="disulfide_" + str(m[0]),
                        disulfides=m[0],
                        remove_existing=remove_existing,
                        repack=repack,
                        scorefxn=scorefxn,
                    )
                    xml.addMover(disulfide)
                    protocol.append(disulfide)

                if relax_cycles:
                    # Create fastrelax mover
                    relax = rosettaScripts.movers.fastRelax(
                        repeats=relax_cycles, scorefxn=sfxn
                    )
                    xml.addMover(relax)
                    protocol.append(relax)
                else:
                    # Turn off more than one structure when relax is not performed
                    nstruct = 1

                # Set protocol
                xml.setProtocol(protocol)

                # Add scorefunction output
                xml.addOutputScorefunction(sfxn)

                # Write XMl protocol file
                xml.write_xml(job_folder + "/xml/" + model + "_" + mutant + ".xml")

                # Create options for minimization protocol
                flags = rosettaScripts.flags(
                    "../../xml/" + model + "_" + mutant + ".xml",
                    nstruct=nstruct,
                    s="../../input_models/" + model + ".pdb",
                    output_silent_file=model + "_" + mutant + ".out",
                )

                # Add relaxation with constraints options and write flags file
                if cst_optimization and relax_cycles:
                    flags.add_relax_cst_options()
                else:
                    flags.add_relax_options()

                # Add path to params files
                if param_files != None:
                    if not os.path.exists(job_folder + "/params"):
                        os.mkdir(job_folder + "/params")
                    if isinstance(param_files, str):
                        param_files = [param_files]
                    for param in param_files:
                        param_name = param.split("/")[-1]
                        shutil.copyfile(param, job_folder + "/params/" + param_name)
                    flags.addOption("in:file:extra_res_path", "../../params")

                flags.write_flags(
                    job_folder + "/flags/" + model + "_" + mutant + ".flags"
                )

                command = "cd " + job_folder + "/output_models/" + model + "\n"
                command += (
                    mpi_command
                    + executable
                    + " @ "
                    + "../../flags/"
                    + model
                    + "_"
                    + mutant
                    + ".flags\n"
                )
                command += "cd ../../..\n"
                jobs.append(command)

        return jobs

    def setUpRosettaOptimization(
        self,
        relax_folder,
        nstruct=_MISSING,
        relax_cycles=5,
        idealize_before_relax=False,
        idealize_only=False,
        cst_files=None,
        mutations=False,
        models=None,
        cst_optimization=True,
        membrane=False,
        membrane_thickness=15,
        param_files=None,
        patch_files=None,
        parallelisation="srun",
        executable="rosetta_scripts.mpi.linuxgccrelease",
        cpus=None,
        skip_finished=True,
        null=False,
        cartesian=False,
        extra_flags=None,
        sugars=False,
        symmetry=False,
        rosetta_path=None,
        ca_constraint=False,
        ligand_chain=None,
        hoh_to_wat=True,
        pdb_output=False,
        interaction_ligand_chains=None,
    ):
        """
        Set up minimizations using Rosetta FastRelax protocol.

        Parameters
        ==========
        relax_folder : str
            Folder path where to place the relax job.
        idealize_before_relax : bool, optional
            Insert an Idealize mover before FastRelax.
        idealize_only : bool, optional
            Run only the Idealize mover and skip FastRelax entirely (mutually exclusive with idealize_before_relax).
        interaction_ligand_chains : list or str, optional
            Chain IDs to report interface scores against (InterfaceAnalyzerMover).
            Not supported with symmetry or membrane.
        """

        # Create minimization job folders
        if not os.path.exists(relax_folder):
            os.mkdir(relax_folder)
        if not os.path.exists(relax_folder + "/input_models"):
            os.mkdir(relax_folder + "/input_models")
        if not os.path.exists(relax_folder + "/flags"):
            os.mkdir(relax_folder + "/flags")
        if not os.path.exists(relax_folder + "/xml"):
            os.mkdir(relax_folder + "/xml")
        if not os.path.exists(relax_folder + "/output_models"):
            os.mkdir(relax_folder + "/output_models")
        if symmetry:
            if not os.path.exists(relax_folder + "/symmetry"):
                os.mkdir(relax_folder + "/symmetry")

        if parallelisation not in ["mpirun", "srun"]:
            raise ValueError("Are you sure about your parallelisation type?")

        if parallelisation == "mpirun" and cpus == None:
            raise ValueError("You must setup the number of cpus when using mpirun")
        if parallelisation == "srun" and cpus != None:
            raise ValueError(
                "CPUs can only be set up when using mpirun parallelisation!"
            )

        if symmetry and interaction_ligand_chains:
            raise ValueError("interaction_ligand_chains is not implemented for symmetry runs.")
        if membrane and interaction_ligand_chains:
            raise ValueError("interaction_ligand_chains is not implemented for membrane runs.")

        nstruct_provided = nstruct is not _MISSING
        if not nstruct_provided:
            nstruct = 1000

        if idealize_only and idealize_before_relax:
            raise ValueError(
                "idealize_only already skips FastRelax; do not combine it with idealize_before_relax."
            )

        if null:
            if nstruct != 1:
                if nstruct_provided:
                    print(
                        f"WARNING: null optimization forces nstruct=1; overriding provided value ({nstruct})."
                    )
            nstruct = 1

        if not idealize_only and cst_optimization and nstruct > 100:
            print(
                "WARNING: A large number of structures (%s) is not necessary when running constrained optimizations!"
                % nstruct
            )
            print("Consider running 100 or less structures.")

        if interaction_ligand_chains is not None:
            if isinstance(interaction_ligand_chains, str):
                interaction_chain_list = [interaction_ligand_chains]
            else:
                interaction_chain_list = list(interaction_ligand_chains)
            if not interaction_chain_list:
                raise ValueError("interaction_ligand_chains cannot be empty.")
        else:
            interaction_chain_list = []

        # Save all models
        self.saveModels(relax_folder + "/input_models", models=models)

        if symmetry and rosetta_path == None:
            raise ValueError(
                "To run relax with symmetry absolute rosetta path must be given to run make_symmdef_file.pl script."
            )

        # Convert any water to WAT name
        if hoh_to_wat:
            for model in self:
                for r in self.structures[model].get_residues():
                    if r.id[0] == 'W':
                        r.resname = 'WAT'

        if symmetry:
            for m in self.models_names:

                # Skip models not in the given list
                if models != None:
                    if model not in models:
                        continue

                ref_chain = symmetry[m][0]
                sym_chains = " ".join(symmetry[m][1:])

                os.system(
                    rosetta_path
                    + "/main/source/src/apps/public/symmetry/make_symmdef_file.pl -p "
                    + relax_folder
                    + "/input_models/"
                    + m
                    + ".pdb -a "
                    + ref_chain
                    + " -i "
                    + sym_chains
                    + " > "
                    + relax_folder
                    + "/symmetry/"
                    + m
                    + ".symm"
                )

        # Check that sequence comparison has been done before adding mutational steps
        if mutations:
            if self.sequence_differences == {}:
                raise ValueError(
                    "Mutations have been enabled but no sequence comparison\
has been carried out. Please run compareSequences() function before setting mutation=True."
                )

        # Check if other cst files have been given.
        if ca_constraint:
            if not cst_files:
                cst_files = {}

        # Reset Rosetta ligand bookkeeping for this run
        self.rosetta_docking_ligands = {}

        # Prepare params folder bookkeeping if needed
        params_output_dir = None
        param_files_list = None
        param_directory_mode = False
        if param_files is not None:
            params_output_dir = Path(relax_folder) / "params"
            params_output_dir.mkdir(exist_ok=True)

            def _read_ligand_name(params_path):
                """Extract ligand residue name from a Rosetta params file."""
                with open(params_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if stripped.startswith("NAME"):
                            parts = stripped.split()
                            if len(parts) >= 2:
                                return parts[1]
                            break
                raise ValueError(
                    f"Could not determine ligand name from params file: {params_path}"
                )

            if isinstance(param_files, (str, os.PathLike)) and Path(
                param_files
            ).is_dir():
                param_directory_mode = True
                param_dir = Path(param_files)
                param_paths = sorted(param_dir.glob("*.params"))
                if not param_paths:
                    raise ValueError(
                        f"No .params files found in directory {param_dir}"
                    )
                # Copy accompanying PDBs if present (e.g., conformers)
                for pdb_file in param_dir.glob("*.pdb"):
                    shutil.copyfile(pdb_file, params_output_dir / pdb_file.name)
                for params_path in param_paths:
                    ligand_name = _read_ligand_name(params_path)
                    destination = params_output_dir / f"{ligand_name}.params"
                    shutil.copyfile(params_path, destination)
                    self.rosetta_docking_ligands[ligand_name] = (
                        f"params/{destination.name}"
                    )
            else:
                if isinstance(param_files, (str, os.PathLike)):
                    param_files_list = [param_files]
                else:
                    param_files_list = list(param_files)

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            if not os.path.exists(relax_folder + "/output_models/" + model):
                os.mkdir(relax_folder + "/output_models/" + model)

            if skip_finished:
                # Check if model has already been calculated and finished
                score_file = (
                    relax_folder + "/output_models/" + model + "/" + model + "_relax.out"
                )
                if os.path.exists(score_file):
                    scores = _readRosettaScoreFile(score_file, skip_empty=True)
                    if not isinstance(scores, type(None)) and scores.shape[0] >= nstruct:
                        continue

            # Validate requested interaction chains exist
            model_chain_ids = {c.id for c in self.structures[model].get_chains()}
            if interaction_chain_list:
                missing = [c for c in interaction_chain_list if c not in model_chain_ids]
                if missing:
                    raise ValueError(
                        f"Ligand chain(s) {missing} not found in model {model}"
                    )

            if ca_constraint:
                if not os.path.exists(relax_folder + "/cst_files"):
                    os.mkdir(relax_folder + "/cst_files")

                if not os.path.exists(relax_folder + "/cst_files/" + model):
                    os.mkdir(relax_folder + "/cst_files/" + model)

                cst_file = (
                    relax_folder + "/cst_files/" + model + "/" + model + "_CA.cst"
                )
                _createCAConstraintFile(self.structures[model], cst_file)

                cst_files.setdefault(model, [])
                cst_files[model].append(cst_file)

            # Create xml minimization protocol
            xml = rosettaScripts.xmlScript()
            protocol = []

            # Create membrane scorefucntion
            if membrane:
                # Create all-atom score function
                weights_file = "mpframework_smooth_fa_2012"
                if cartesian:
                    weights_file += "_cart"

                sfxn = rosettaScripts.scorefunctions.new_scorefunction(
                    "mpframework_smooth_fa_2012", weights_file=weights_file
                )

                # Add constraint weights to membrane score function
                if cst_files != None:
                    reweights = (
                        ("chainbreak", 1.0),
                        ("coordinate_constraint", 1.0),
                        ("atom_pair_constraint", 1.0),
                        ("angle_constraint", 1.0),
                        ("dihedral_constraint", 1.0),
                        ("res_type_constraint", 1.0),
                        ("metalbinding_constraint", 1.0),
                    )

                    for rw in reweights:
                        sfxn.addReweight(rw[0], rw[1])

            # Create all-atom scorefucntion
            else:
                score_fxn_name = "ref2015"

                if cartesian:
                    score_fxn_name += "_cart"

                # Check if constraints are given
                if cst_files != None:
                    score_fxn_name += "_cst"

                # Create all-atom score function
                sfxn = rosettaScripts.scorefunctions.new_scorefunction(
                    score_fxn_name, weights_file=score_fxn_name
                )
            xml.addScorefunction(sfxn)

            # Detect symmetry if specified
            if symmetry:
                # detect_symmetry = rosettaScripts.movers.DetectSymmetry(subunit_tolerance=1, plane_tolerance=1)
                setup_symmetry = rosettaScripts.movers.SetupForSymmetry(
                    definition="../../symmetry/" + model + ".symm"
                )
                xml.addMover(setup_symmetry)
                protocol.append(setup_symmetry)

            # Create mutation movers if needed
            if mutations:
                if self.sequence_differences[model]["mutations"] != {}:
                    for m in self.sequence_differences[model]["mutations"]:
                        mutate = rosettaScripts.movers.mutate(
                            name="mutate_" + str(m[0]),
                            target_residue=m[0],
                        new_residue=_one_to_three(m[1]),
                        )
                        xml.addMover(mutate)
                        protocol.append(mutate)

            # Add constraint mover if constraint file is given.
            if cst_files != None:
                if model not in cst_files:
                    raise ValueError(
                        "Model %s is not in the cst_files dictionary!" % model
                    )

                if isinstance(cst_files[model], str):
                    cst_files[model] = [cst_files[model]]

                if not os.path.exists(relax_folder + "/cst_files"):
                    os.mkdir(relax_folder + "/cst_files")

                if not os.path.exists(relax_folder + "/cst_files/" + model):
                    os.mkdir(relax_folder + "/cst_files/" + model)

                for cst_file in cst_files[model]:

                    cst_file_name = cst_file.split("/")[-1]

                    if not os.path.exists(
                        relax_folder + "/cst_files/" + model + "/" + cst_file_name
                    ):
                        shutil.copyfile(
                            cst_file,
                            relax_folder + "/cst_files/" + model + "/" + cst_file_name,
                        )

                    set_cst = rosettaScripts.movers.constraintSetMover(
                        add_constraints=True,
                        cst_file="../../cst_files/" + model + "/" + cst_file_name,
                    )
                xml.addMover(set_cst)
                protocol.append(set_cst)

            if membrane:
                add_membrane = rosettaScripts.rosetta_MP.movers.addMembraneMover()
                xml.addMover(add_membrane)
                protocol.append(add_membrane)

                init_membrane = (
                    rosettaScripts.rosetta_MP.movers.membranePositionFromTopologyMover()
                )
                xml.addMover(init_membrane)
                protocol.append(init_membrane)

            if idealize_before_relax or idealize_only:
                idealize_mover = rosettaScripts.movers.idealize()
                xml.addMover(idealize_mover)
                if not null:
                    protocol.append(idealize_mover)

            relax = None
            if not idealize_only:
                # Create fastrelax mover
                relax = rosettaScripts.movers.fastRelax(
                    repeats=relax_cycles, scorefxn=sfxn
                )
                xml.addMover(relax)

                if not null:
                    protocol.append(relax)

            # Add interface analysis movers for requested ligand chains
            if interaction_chain_list:
                for chain_id in interaction_chain_list:
                    iam = rosettaScripts.movers.interfaceAnalyzerMover(
                        name=f"interface_anl_{chain_id}",
                        scorefxn=sfxn.name,
                        ligandchain=chain_id,
                        scorefile_reporting_prefix=f"interface_score_{chain_id}",
                    )
                    xml.addMover(iam)
                    protocol.append(iam)

            # Set protocol
            xml.setProtocol(protocol)

            # Add scorefunction output
            xml.addOutputScorefunction(sfxn)

            # Write XMl protocol file
            xml.write_xml(relax_folder + "/xml/" + model + "_relax.xml")

            if symmetry:
                input_model = model + "_INPUT.pdb"
            else:
                input_model = model + ".pdb"

            # Create options for minimization protocol
            if pdb_output:
                output_silent_file = None
            else:
                output_silent_file = model + "_relax.out"

            flags = rosettaScripts.flags(
                "../../xml/" + model + "_relax.xml",
                nstruct=nstruct,
                s="../../input_models/" + input_model,
                output_silent_file=output_silent_file
            )

            # Add extra flags
            if extra_flags != None:
                for o in extra_flags:
                    if isinstance(o, tuple):
                        flags.addOption(*o)
                    else:
                        flags.addOption(o)

            # Add relaxation with constraints options and write flags file
            if not idealize_only:
                if cst_optimization:
                    flags.add_relax_cst_options()
                else:
                    flags.add_relax_options()

            # Add path to params files
            if param_files != None:

                for r in self.structures[model].get_residues():
                    if r.resname == 'NMA':
                        _copyScriptFile(relax_folder+"/params", 'NMA.params', subfolder='rosetta_params', path='prepare_proteins', hidden=False)

                patch_entries = []
                if param_directory_mode:
                    pass
                else:
                    for param in param_files_list:
                        param_path = Path(param)
                        destination = params_output_dir / param_path.name
                        if not destination.exists():
                            shutil.copyfile(param_path, destination)
                        if not param_path.name.endswith(".params"):
                            patch_entries.append("../../params/" + param_path.name)

                flags.addOption("in:file:extra_res_path", "../../params")
                if patch_entries:
                    flags.addOption("in:file:extra_patch_fa", " ".join(patch_entries))

            if membrane:
                flags.addOption("mp::setup::spans_from_structure", "true")
                if not idealize_only:
                    flags.addOption("relax:constrain_relax_to_start_coords")

            if sugars:
                flags.addOption("include_sugars")
                flags.addOption("alternate_3_letter_codes", "pdb_sugar")
                flags.addOption("write_glycan_pdb_codes")
                flags.addOption("auto_detect_glycan_connections")
                flags.addOption("maintain_links")

            flags.write_flags(relax_folder + "/flags/" + model + "_relax.flags")

            # Create and append execution command
            command = "cd " + relax_folder + "/output_models/" + model + "\n"
            if parallelisation == "mpirun":
                if cpus == 1:
                    command += (
                        executable + " @ " + "../../flags/" + model + "_relax.flags\n"
                    )
                else:
                    command += (
                        "mpirun -np "
                        + str(cpus)
                        + " "
                        + executable
                        + " @ "
                        + "../../flags/"
                        + model
                        + "_relax.flags\n"
                    )
            else:
                command += (
                    "srun "
                    + executable
                    + " @ "
                    + "../../flags/"
                    + model
                    + "_relax.flags\n"
                )
            command += "cd ../../..\n"
            jobs.append(command)

        return jobs

    def setUpMembranePositioning(self, job_folder, membrane_thickness=15, models=None):
        """ """
        # Create minimization job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")
        if not os.path.exists(job_folder + "/flags"):
            os.mkdir(job_folder + "/flags")
        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Save all models
        self.saveModels(job_folder + "/input_models", models=models)

        # Copy embeddingToMembrane.py script
        _copyScriptFile(job_folder, "embeddingToMembrane.py")

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            if not os.path.exists(job_folder + "/output_models/" + model):
                os.mkdir(job_folder + "/output_models/" + model)

            flag_file = job_folder + "/flags/mp_span_from_pdb_" + model + ".flags"
            with open(flag_file, "w") as flags:
                flags.write("-mp::thickness " + str(membrane_thickness) + "\n")
                flags.write("-s model.pdb\n")
                flags.write("-out:path:pdb .\n")

            # flag_file = job_folder+'/flags/mp_transform_'+model+'.flags'
            # with open(flag_file, 'w') as flags:
            #     flags.write('-s ../../input_models/'+model+'.pdb\n')
            #     flags.write('-mp:transform:optimize_embedding true\n')
            #     flags.write('-mp:setup:spanfiles '+model+'.span\n')
            #     flags.write('-out:no_nstruct_label\n')

            command = "cd " + job_folder + "/output_models/" + model + "\n"
            command += "cp ../../input_models/" + model + ".pdb model.pdb\n"
            command += (
                "mp_span_from_pdb.linuxgccrelease @ ../../flags/mp_span_from_pdb_"
                + model
                + ".flags\n"
            )
            # command += 'rm model.pdb \n'
            command += "mv model.pdb " + model + ".pdb\n"
            command += "mv model.span " + model + ".span\n"
            # command += 'mp_transform.linuxgccrelease @ ../../flags/mp_transform_'+model+'.flags\n'
            # command += 'python ../../._embeddingToMembrane.py'+' '+model+'.pdb\n'
            command += "cd ../../..\n"
            jobs.append(command)

        return jobs

    def addMissingLoops(
        self, job_folder, nstruct=1, sfxn="ref2015", param_files=None, idealize=True
    ):
        """
        Create a Rosetta loop optimization protocol for missing loops in the structure.

        Parameters
        ==========
        job_folder : str
            Loop modeling calculation folder.
        """

        # Create minimization job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")
        if not os.path.exists(job_folder + "/flags"):
            os.mkdir(job_folder + "/flags")
        if not os.path.exists(job_folder + "/xml"):
            os.mkdir(job_folder + "/xml")
        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Save all models
        self.saveModels(job_folder + "/input_models")

        # Check that sequence comparison has been done before checking missing loops
        if self.sequence_differences == {}:
            raise ValueError(
                "No sequence comparison has been carried out. Please run \
compareSequences() function before adding missing loops."
            )

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Check that model has missing loops
            if self.sequence_differences[model]["missing_loops"] != []:

                missing_loops = self.sequence_differences[model]["missing_loops"]

                for loop in missing_loops:

                    loop_name = str(loop[0]) + "_" + str(loop[1])

                    if not os.path.exists(job_folder + "/output_models/" + model):
                        os.mkdir(job_folder + "/output_models/" + model)

                    if not os.path.exists(
                        job_folder + "/output_models/" + model + "/" + loop_name
                    ):
                        os.mkdir(
                            job_folder + "/output_models/" + model + "/" + loop_name
                        )

                    # Create xml minimization protocol
                    xml = rosettaScripts.xmlScript()
                    protocol = []

                    # Create score function

                    scorefxn = rosettaScripts.scorefunctions.new_scorefunction(
                        sfxn, weights_file=sfxn
                    )
                    # Add loop remodel protocol
                    if len(loop[1]) == 1:
                        hanging_residues = 3
                    elif len(loop[1]) == 2:
                        hanging_residues = 2
                    else:
                        hanging_residues = 1
                    loop_movers = rosettaScripts.loop_modeling.loopRebuild(
                        xml,
                        loop[0],
                        loop[1],
                        scorefxn=sfxn,
                        hanging_residues=hanging_residues,
                    )
                    for m in loop_movers:
                        protocol.append(m)

                    # Add idealize step
                    if idealize:
                        idealize = rosettaScripts.movers.idealize()
                        xml.addMover(idealize)
                        protocol.append(idealize)

                    # Set protocol
                    xml.setProtocol(protocol)

                    # Add scorefunction output
                    xml.addOutputScorefunction(scorefxn)
                    # Write XMl protocol file
                    xml.write_xml(
                        job_folder + "/xml/" + model + "_" + loop_name + ".xml"
                    )

                    # Create options for minimization protocol
                    output_silent = (
                        "output_models/"
                        + model
                        + "/"
                        + loop_name
                        + "/"
                        + model
                        + "_"
                        + loop_name
                        + ".out"
                    )
                    flags = rosettaScripts.flags(
                        "xml/" + model + "_" + loop_name + ".xml",
                        nstruct=nstruct,
                        s="input_models/" + model + ".pdb",
                        output_silent_file=output_silent,
                    )

                    # Add path to params files
                    if param_files != None:
                        if not os.path.exists(job_folder + "/params"):
                            os.mkdir(job_folder + "/params")

                        if isinstance(param_files, str):
                            param_files = [param_files]
                        for param in param_files:
                            param_name = param.split("/")[-1]
                            shutil.copyfile(param, job_folder + "/params/" + param_name)
                        flags.addOption("in:file:extra_res_path", "params")

                    # Write flags file
                    flags.write_flags(
                        job_folder + "/flags/" + model + "_" + loop_name + ".flags"
                    )

                    # Create and append execution command
                    command = "cd " + job_folder + "\n"
                    command += (
                        "srun rosetta_scripts.mpi.linuxgccrelease @ "
                        + "flags/"
                        + model
                        + "_"
                        + loop_name
                        + ".flags\n"
                    )
                    command += "cd ..\n"

                    jobs.append(command)

        return jobs

    def setUpPrepwizardOptimization(
        self,
        prepare_folder,
        pH=7.0,
        epik_pH=False,
        samplewater=False,
        models=None,
        epik_pHt=False,
        remove_hydrogens=False,
        delwater_hbond_cutoff=False,
        fill_loops=False,
        protonation_states=None,
        noepik=False,
        mae_input=False,
        noprotassign=False,
        use_new_version=False,
        replace_symbol=None,
        captermini=False,
        keepfarwat=False,
        skip_finished=False,
        **kwargs,
    ):
        """
        Set up an structure optimization with the Schrodinger Suite prepwizard.

        Parameters
        ==========
        prepare_folder : str
            Folder name for the prepwizard optimization.
        """

        # Create prepare job folders
        if not os.path.exists(prepare_folder):
            os.mkdir(prepare_folder)
        if not os.path.exists(prepare_folder + "/input_models"):
            os.mkdir(prepare_folder + "/input_models")
        if not os.path.exists(prepare_folder + "/output_models"):
            os.mkdir(prepare_folder + "/output_models")

        # Save all input models
        self.saveModels(
            prepare_folder + "/input_models",
            convert_to_mae=mae_input,
            remove_hydrogens=remove_hydrogens,
            replace_symbol=replace_symbol,
            models=models,
        )  # **kwargs)

        # Generate jobs
        jobs = []
        for model in self.models_names:

            if models != None and model not in models:
                continue

            if replace_symbol:
                model_name = model.replace(replace_symbol[0], replace_symbol[1])
            else:
                model_name = model

            output_path = (
                prepare_folder
                + "/output_models/"
                + model_name
                + "/"
                + model_name
                + ".pdb"
            )
            if skip_finished and os.path.exists(output_path):
                continue

            if fill_loops:
                if model not in self.target_sequences:
                    raise ValueError(
                        "Target sequence for model %s was not given. First\
make sure of reading the target sequences with the function readTargetSequences()"
                        % model
                    )
                sequence = {}
                sequence[model] = self.target_sequences[model]
                fasta_file = prepare_folder + "/input_models/" + model_name + ".fasta"
                alignment.writeFastaFile(sequence, fasta_file)

            # Create model output folder
            output_folder = prepare_folder + "/output_models/" + model_name
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            if fill_loops:
                command = "cd " + prepare_folder + "/input_models/\n"
                command += "pwd=$(pwd)\n"
                command += "cd ../output_models/" + model_name + "\n"
            else:
                command = "cd " + output_folder + "\n"

            command += '"${SCHRODINGER}/utilities/prepwizard" '
            if mae_input:
                command += "../../input_models/" + model_name + ".mae "
            else:
                command += "../../input_models/" + model_name + ".pdb "
            command += model_name + ".pdb "
            command += "-fillsidechains "
            command += "-disulfides "
            if keepfarwat:
                command += "-keepfarwat "
            if fill_loops:
                command += "-fillloops "
                command += '-fasta_file "$pwd"/' + model_name + ".fasta "
            if captermini:
                command += "-captermini "
            if remove_hydrogens:
                command += "-rehtreat "
            if noepik:
                command += "-noepik "
            if noprotassign:
                command += "-noprotassign "
            else:
                if epik_pH:
                    command += "-epik_pH " + str(pH) + " "
                if epik_pHt:
                    command += "-epik_pHt " + str(epik_pHt) + " "
            command += "-propka_pH " + str(pH) + " "
            command += "-f OPLS_2005 "
            command += "-rmsd 0.3 "
            if samplewater:
                command += "-samplewater "
            if delwater_hbond_cutoff:
                command += "-delwater_hbond_cutoff " + str(delwater_hbond_cutoff) + " "

            if not isinstance(protonation_states, type(None)):
                for ps in protonation_states[model]:
                    if use_new_version:
                        command += "-force " + str(ps[0]) + " " + str(ps[1]) + " "
                    else:
                        command += "-force " + str(ps[0]) + " " + str(ps[1]) + " "

            command += "-JOBNAME " + model_name + " "
            command += "-HOST localhost:1 "
            command += "-WAIT\n"
            command += "cd ../../..\n"
            jobs.append(command)

        return jobs

    def setUpDockingGrid(
        self,
        grid_folder,
        center_atoms,
        innerbox=(10, 10, 10),
        outerbox=(30, 30, 30),
        useflexmae=True,
        peptide=False,
        mae_input=True,
        cst_positions=None,
        models=None,
        exclude_models=None,
        skip_finished=False,
    ):
        """
        Generate Glide grid input files and return the shell commands required to
        launch each grid calculation.

        Args:
            grid_folder (str): Directory where input/output subfolders and grid
                files will be created.
            center_atoms (dict[str, tuple]): Mapping from model name to either
                (x, y, z) coordinates or a `(chain_id, residue_index, atom_name)`
                triplet whose coordinates are used as the grid center.
            innerbox (tuple[int, int, int]): Dimensions of the inner search box
                in Glide grid units (angstroms).
            outerbox (tuple[int, int, int]): Dimensions of the outer search box
                in Glide grid units (angstroms).
            useflexmae (bool): Whether to request a flexible receptor MAE in the
                generated input file.
            peptide (bool): Flag the receptor as a peptide during grid
                generation.
            mae_input (bool): If True, receptors are written and referenced as
                MAE files; otherwise PDB files are used.
            cst_positions (dict[str, list[tuple]] | None): Optional mapping from
                model name to an iterable of `((chain_id, residue_index,
                atom_name), radius)` tuples used to generate Glide positional
                constraints.
            models (list[str] | str | None): Optional subset of model identifiers
                to process.
            exclude_models (list[str] | str | None): Optional collection of model
                identifiers that should be skipped.
            skip_finished (bool): When True, skip models that already have a
                corresponding grid output file in `grid_folder/output_models`.

        Returns:
            list[str]: Shell command strings that execute Glide grid
            calculations for the selected models.

        Raises:
            ValueError: If a requested grid center or positional constraint atom
            cannot be found, or if inner/outer box dimensions are not integers.
        """

        # Create grid job folders
        if not os.path.exists(grid_folder):
            os.mkdir(grid_folder)

        if not os.path.exists(grid_folder + "/input_models"):
            os.mkdir(grid_folder + "/input_models")

        if not os.path.exists(grid_folder + "/grid_inputs"):
            os.mkdir(grid_folder + "/grid_inputs")

        if not os.path.exists(grid_folder + "/output_models"):
            os.mkdir(grid_folder + "/output_models")

        if isinstance(models, str):
            models = [models]

        if isinstance(exclude_models, str):
            exclude_models = [exclude_models]

        # Save all input models
        self.saveModels(grid_folder + "/input_models", convert_to_mae=mae_input, models=models)

        # Check that inner and outerbox values are given as integers
        for v in innerbox:
            if type(v) != int:
                raise ValueError("Innerbox values must be given as integers")
        for v in outerbox:
            if type(v) != int:
                raise ValueError("Outerbox values must be given as integers")

        # Create grid input files
        jobs = []
        for model in self.models_names:

            if models and model not in models:
                continue

            if exclude_models and model in exclude_models:
                continue

            # Check if output grid exists
            output_path = grid_folder + "/output_models/" + model + ".zip"
            if skip_finished and os.path.exists(output_path):
                continue

            if all([isinstance(x, (float, int)) for x in center_atoms[model]]):
                x = float(center_atoms[model][0])
                y = float(center_atoms[model][1])
                z = float(center_atoms[model][2])

            else:
                # Get coordinates of center residue
                chainid = center_atoms[model][0]
                resid = center_atoms[model][1]
                atom_name = center_atoms[model][2]

                x = None
                for c in self.structures[model].get_chains():
                    if c.id == chainid:
                        for r in c.get_residues():
                            if r.id[1] == resid:
                                for a in r.get_atoms():
                                    if a.name == atom_name:
                                        x = a.coord[0]
                                        y = a.coord[1]
                                        z = a.coord[2]

            if cst_positions != None:

                cst_x = {}
                cst_y = {}
                cst_z = {}

                # Convert to a list of only one position cst is given
                if isinstance(cst_positions[model], tuple):
                    cst_positions[model] = [cst_positions[model]]

                for i, position in enumerate(cst_positions[model]):

                    # Get coordinates of center residue
                    chainid = position[0][0]
                    resid = position[0][1]
                    atom_name = position[0][2]

                    for c in self.structures[model].get_chains():
                        if c.id == chainid:
                            for r in c.get_residues():
                                if r.id[1] == resid:
                                    for a in r.get_atoms():
                                        if a.name == atom_name:
                                            cst_x[i + 1] = a.coord[0]
                                            cst_y[i + 1] = a.coord[1]
                                            cst_z[i + 1] = a.coord[2]

            # Check if any atom center was found.
            if x == None:
                raise ValueError("Given atom center not found for model %s" % model)

            # Check if any atom center was found.
            if cst_positions and cst_x == {}:
                raise ValueError("Given atom constraint not found for model %s" % model)

            # Write grid input file
            with open(grid_folder + "/grid_inputs/" + model + ".in", "w") as gif:
                gif.write(
                    "GRID_CENTER %.14f, %.14f, %.14f\n"
                    % (
                        x,
                        y,
                        z,
                    )
                )
                gif.write("GRIDFILE " + model + ".zip\n")
                gif.write("INNERBOX %s, %s, %s\n" % innerbox)
                gif.write("OUTERBOX %s, %s, %s\n" % outerbox)

                if cst_positions != None:
                    parts = []
                    for i, position in enumerate(cst_positions[model]):
                        parts.append(
                            "position%s %.14f %.14f %.14f %.14f"
                            % (
                                i + 1,
                                cst_x[i + 1],
                                cst_y[i + 1],
                                cst_z[i + 1],
                                position[-1],
                            )
                        )
                    if parts:
                        gif.write(
                            "POSIT_CONSTRAINTS "
                            + ", ".join(f'"{p}"' for p in parts)
                            + "\n"
                        )

                if mae_input:
                    gif.write("RECEP_FILE %s\n" % ("../input_models/" + model + ".mae"))
                else:
                    gif.write("RECEP_FILE %s\n" % ("../input_models/" + model + ".pdb"))
                if peptide:
                    gif.write("PEPTIDE True\n")
                if useflexmae:
                    gif.write("USEFLEXMAE YES\n")

            command = "cd " + grid_folder + "/output_models\n"

            # Add grid generation command
            command += '"${SCHRODINGER}/glide" '
            command += "../grid_inputs/" + model + ".in" + " "
            command += "-OVERWRITE "
            command += "-HOST localhost "
            command += "-TMPLAUNCHDIR "
            command += "-WAIT\n"
            command += "cd ../..\n"

            jobs.append(command)

        return jobs

    def setUpGlideDocking(
        self,
        docking_folder,
        grids_folder,
        ligands_folder,
        models=None,
        poses_per_lig=100,
        precision="SP",
        use_ligand_charges=False,
        energy_by_residue=False,
        use_new_version=False,
        cst_fragments=None,
        skip_finished=None,
        only_ligands=None,
        failed_pairs=None,
    ):
        """
        Build Glide docking input files for every grid/ligand combination and
        return the commands needed to execute those dockings.

        Args:
            docking_folder (str): Base directory where docking inputs and
                results will be written.
            grids_folder (str): Directory that contains previously generated
                Glide grid ZIP archives under `output_models/`.
            ligands_folder (str): Directory containing ligand structures in MAE
                format.
            models (list[str] | str | None): Optional set of model identifiers to
                consider; others will be skipped.
            poses_per_lig (int): Number of poses Glide should keep for each
                ligand.
            precision (str): Glide precision mode (e.g., `"SP"` or `"XP"`).
            use_ligand_charges (bool): If True, request the use of ligand MAE
                charges during docking.
            energy_by_residue (bool): If True, write per-residue interaction
                energies.
            use_new_version (bool): Reserved flag for downstream consumers that
                require alternate command generation (currently unused).
            cst_fragments (dict[str, dict[str, tuple | list[tuple]]] | None):
                Optional mapping from grid name to ligand name to either a single
                constraint tuple or an iterable of tuples describing positional
                constraints in the form `(smarts, feature_index, include_flag)`.
            skip_finished (bool | None): When True, skip docking runs that
                already produced a `_pv.maegz` output file.
            only_ligands (list[str] | str | None): Optional subset of ligand
                names to include from `ligands_folder`.
            failed_pairs (Iterable[tuple[str, str]] | None): Optional iterable
                of `(protein, ligand)` identifiers produced by
                `analyseDocking(..., return_failed=True)`; when provided, only
                those combinations are prepared (after applying the other
                filters).

        Example:
            >>> cst_fragments = {
            ...     "ProteinA": {
            ...         "ligand1": [
            ...             ("O=C[O-]", 2, True),
            ...             ("[H]C([H])[H]", 2, True),
            ...         ],
            ...         "ligand2": ("c1ccccc1", 4, False),
            ...     }
            ... }
            >>> jobs = models.setUpGlideDocking(
            ...     "docking_runs", "grids", "ligands", cst_fragments=cst_fragments
            ... )
            This generates a Glide input where the `USE_CONS` line references two
            grid positions and the `PATTERN1` entries match the provided SMARTS
            strings and indices.

        Returns:
            list[str]: Shell command strings that execute Glide docking jobs for
            the prepared grid/ligand combinations.
        """

        if isinstance(only_ligands, str):
            only_ligands = [only_ligands]

        if isinstance(models, str):
            models = [models]

        failed_map = None
        if failed_pairs:
            failed_map = {}
            for pair in failed_pairs:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError(
                        "Each item in failed_pairs must be a (protein, ligand) pair."
                    )
                protein, ligand = pair
                protein = str(protein)
                ligand = str(ligand)
                failed_map.setdefault(protein, set()).add(ligand)

        # Create docking job folders
        if not os.path.exists(docking_folder):
            os.mkdir(docking_folder)

        if not os.path.exists(docking_folder + "/input_models"):
            os.mkdir(docking_folder + "/input_models")

        if not os.path.exists(docking_folder + "/output_models"):
            os.mkdir(docking_folder + "/output_models")

        # Save all input models
        self.saveModels(docking_folder + "/input_models", models=models)

        # Read paths to grid files
        grids_paths = {}
        for f in os.listdir(grids_folder + "/output_models"):
            if f.endswith(".zip"):
                name = f.replace(".zip", "")
                grids_paths[name] = grids_folder + "/output_models/" + f

        # Read paths to substrates
        substrates_paths = {}
        for f in os.listdir(ligands_folder):
            if f.endswith(".mae"):
                name = f.replace(".mae", "")

                if only_ligands and name not in only_ligands:
                    continue

                substrates_paths[name] = ligands_folder + "/" + f

        # Set up docking jobs
        jobs = []
        for grid in grids_paths:

            # Skip if models are given and not in models
            if models:
                if grid not in models:
                    continue
            if failed_map is not None and grid not in failed_map:
                continue

            # Create ouput folder
            if not os.path.exists(docking_folder + "/output_models/" + grid):
                os.mkdir(docking_folder + "/output_models/" + grid)

            for substrate in substrates_paths:

                if failed_map is not None and substrate not in failed_map.get(grid, set()):
                    continue

                output_path = docking_folder+"/output_models/"+grid+'/'+grid+"_"+ substrate+'_pv.maegz'
                if skip_finished and os.path.exists(output_path):
                    continue

                # Create glide dock input
                with open(
                    docking_folder
                    + "/output_models/"
                    + grid
                    + "/"
                    + grid
                    + "_"
                    + substrate
                    + ".in",
                    "w",
                ) as dif:
                    dif.write("GRIDFILE GRID_PATH/" + grid + ".zip\n")
                    dif.write("LIGANDFILE ../../../%s\n" % substrates_paths[substrate])
                    dif.write("POSES_PER_LIG %s\n" % poses_per_lig)
                    if use_ligand_charges:
                        dif.write("LIG_MAECHARGES true\n")
                    dif.write("PRECISION %s\n" % precision)
                    if energy_by_residue:
                        dif.write("WRITE_RES_INTERACTION true\n")
                    # Constraints
                    if cst_fragments != None:

                        if isinstance(cst_fragments[grid][substrate], tuple):
                            cst_fragments[grid][substrate] = [
                                cst_fragments[grid][substrate]
                            ]

                        fragments = cst_fragments[grid][substrate]
                        if fragments:
                            dif.write("[CONSTRAINT_GROUP:1]\n")
                            use_cons = ", ".join(
                                [f"position{k}:{k}" for k in range(1, len(fragments) + 1)]
                            )
                            dif.write("\tUSE_CONS  " + use_cons + "\n")
                            dif.write("\tNREQUIRED_CONS ALL\n")

                            for i, fragment in enumerate(fragments):
                                dif.write(f"[FEATURE:{i + 1}]\n")
                                dif.write(
                                    '\tPATTERN1 "'
                                    + fragment[0]
                                    + " "
                                    + str(fragment[1])
                                )
                                if fragment[2]:
                                    dif.write(" include")
                                dif.write('"\n')

                # Create commands
                command = "cd " + docking_folder + "/output_models/" + grid + "\n"

                # Schrodinger has problem with relative paths to the grid files
                # This is a quick fix for that (not elegant, but works).
                command += "cwd=$(pwd)\n"
                grid_folder = "/".join(grids_paths[grid].split("/")[:-1])
                command += "cd ../../../%s\n" % grid_folder
                command += "gd=$(pwd)\n"
                command += "cd $cwd\n"
                command += 'sed -i "s@GRID_PATH@$gd@" %s \n' % (
                    grid + "_" + substrate + ".in"
                )

                # Add docking command
                command += '"${SCHRODINGER}/glide" '
                command += grid + "_" + substrate + ".in" + " "
                command += "-OVERWRITE "
                command += "-adjust "
                command += "-HOST localhost:1 "
                command += "-TMPLAUNCHDIR "
                command += "-WAIT\n"
                command += "cd ../../..\n"
                jobs.append(command)

        return jobs

    def computeModelContacts(
        self,
        chains: Optional[Union[str, Iterable[str], Dict[str, Iterable[str]]]] = None,
        *,
        models: Optional[Union[str, Iterable[str]]] = None,
        mode: Literal["chains", "ligands", "both"] = "both",
        cutoff: float = 5.0,
        second_shell: Optional[float] = None,
        atom_scope: Literal["all", "heavy", "backbone", "sidechain"] = "heavy",
        group_by: Literal["residue", "atom", "chain"] = "residue",
        exclude_query_intra: bool = True,
        query_residues: Optional[Any] = None,
        include: Optional[Dict[str, Any]] = None,
        ligand_filters: Optional[Dict[str, Any]] = None,
        chain_filters: Optional[Dict[str, Any]] = None,
        altloc_mode: Literal["first", "best_occ"] = "first",
        print_chain_summary: bool = False,
        only_residue_dict: bool = False,
        exclude_bb_contacts: bool = False,
        classify_bb_sc: bool = True,
        filter_contact_classes: Optional[Set[str]] = None,
        split_counts_by_class: bool = False,
    ):
        """
        Compute neighbour contacts for the selected models and cache them.

        This is a thin convenience wrapper around
        :func:`prepare_proteins.analysis.find_neighbours_in_pdb` that applies the
        same contact-detection logic to every selected model and stores the results
        on ``self.models_contacts``.

        By default all available models and all polymer chains in each model are
        analysed. You can restrict the calculation to specific models via
        ``models`` and to specific chains via ``chains``.

        Args:
            chains: Optional chain selector restricting which chains are analysed.
                Can be a single chain ID (``\"A\"``), an iterable of chain IDs
                (e.g. ``[\"A\", \"B\"]``), or a mapping of model name to iterables
                of chain IDs (for per-model selection). If ``None``, all polymer
                chains in each model are used.
            models: Optional model selector. Either a single model name or an
                iterable of model names. If ``None``, all models in
                ``self.models_names`` are analysed.
            mode: Contact mode passed through to
                :func:`prepare_proteins.analysis.find_neighbours_in_pdb`, controlling
                whether chain–chain, ligand, or both types of contacts are included.
            cutoff: Distance cutoff (in Å) for defining direct contacts.
            second_shell: Optional second-shell cutoff (in Å); if provided, a
                second, looser distance shell is also computed.
            atom_scope: Atom selection used when computing distances (all atoms,
                heavy atoms only, backbone, or side chains).
            group_by: Granularity of the returned contacts (per residue, per atom,
                or per chain).
            exclude_query_intra: If ``True``, intra-chain contacts within the query
                selection are excluded from the results.
            query_residues: Optional residue selection used as explicit query set;
                passed through to :func:`find_neighbours_in_pdb`.
            include: Optional additional atom / residue selection rules; forwarded
                to :func:`find_neighbours_in_pdb`.
            ligand_filters: Optional filters restricting which ligands are included
                in the contact analysis.
            chain_filters: Optional filters restricting which protein chains are
                considered as potential neighbours.
            altloc_mode: Strategy for handling alternate locations in the PDB file
                (e.g. first altloc vs. highest occupancy).
            print_chain_summary: If ``True``, print a short per-chain summary for
                each processed model.
            only_residue_dict: If ``True``, return only residue-level contact
                dictionaries instead of the full result object.
            exclude_bb_contacts: If ``True``, backbone-only contacts are excluded.
            classify_bb_sc: If ``True``, contacts are classified as backbone or
                side-chain contacts.
            filter_contact_classes: Optional set of contact class labels to retain
                in the output.
            split_counts_by_class: If ``True``, contact counts are split by contact
                class instead of aggregated.

        Returns:
            dict[str, Any]: Mapping of model name to the contact-analysis result
            returned by :func:`find_neighbours_in_pdb` for that model.
        """
        available_models = list(self.models_names)
        if models is None:
            selected_models = available_models
        else:
            if isinstance(models, str):
                requested_models = [str(models)]
            else:
                requested_models = [str(m) for m in models]
            missing_models = sorted(set(requested_models) - set(self.models_paths))
            if missing_models:
                raise ValueError(f"Models not found: {', '.join(missing_models)}")
            selected_models = [m for m in available_models if m in requested_models]
            for model in requested_models:
                if model not in selected_models:
                    selected_models.append(model)

        if not selected_models:
            raise ValueError("No models available for contact analysis.")

        chain_map = self._resolve_chain_selection(chains=chains, models=selected_models)

        results: Dict[str, Any] = {}
        for model in selected_models:
            pdb_path = self.models_paths.get(model)
            if pdb_path is None:
                raise ValueError(f"PDB path for model '{model}' not available.")
            model_chains = chain_map.get(model, [])
            if not model_chains:
                raise ValueError(f"No chains available for model '{model}'.")
            results[model] = find_neighbours_in_pdb(
                pdb_path,
                model_chains,
                mode=mode,
                cutoff=cutoff,
                second_shell=second_shell,
                atom_scope=atom_scope,
                group_by=group_by,
                exclude_query_intra=exclude_query_intra,
                query_residues=query_residues,
                include=include,
                ligand_filters=ligand_filters,
                chain_filters=chain_filters,
                altloc_mode=altloc_mode,
                print_chain_summary=print_chain_summary,
                only_residue_dict=only_residue_dict,
                exclude_bb_contacts=exclude_bb_contacts,
                classify_bb_sc=classify_bb_sc,
                filter_contact_classes=filter_contact_classes,
                split_counts_by_class=split_counts_by_class,
            )

        self.models_contacts = results
        return results

    def buildMutationAnalyzer(
        self,
        wild_type: str,
        *,
        chains: Optional[Union[str, Iterable[str], Dict[str, Iterable[str]]]] = None,
        only_models: Optional[Union[str, Iterable[str]]] = None,
        exclude_models: Optional[Union[str, Iterable[str]]] = None,
        alignment_params: Optional[Dict[str, float]] = None,
        residue_annotations: Optional[Union[Dict[Any, str], pd.DataFrame]] = None,
        contact_results: Optional[Dict[str, Any]] = None,
        auto_classify_locations: bool = True,
        core_contact_threshold: int = 12,
        surface_contact_threshold: int = 4,
        pocket_neighbor_kinds: Optional[Iterable[str]] = None,
    ) -> MutationVariabilityAnalyzer:
        """Convenience wrapper to instantiate :class:`MutationVariabilityAnalyzer`.

        Parameters mirror the analyzer constructor. Use ``only_models`` and
        ``exclude_models`` to restrict the set of designs included in the
        mutational analysis. Pass ``contact_results`` to override cached
        neighbour data (defaults to ``self.models_contacts``).
        """

        if contact_results is not None:
            analysis_contacts = contact_results
        else:
            analysis_contacts = self.models_contacts
            if not analysis_contacts:
                self.computeModelContacts(chains=chains)
                analysis_contacts = self.models_contacts
        return MutationVariabilityAnalyzer(
            self,
            wild_type,
            only_models=only_models,
            exclude_models=exclude_models,
            chains=chains,
            alignment_params=alignment_params,
            residue_annotations=residue_annotations,
            contact_results=analysis_contacts,
            auto_classify_locations=auto_classify_locations,
            core_contact_threshold=core_contact_threshold,
            surface_contact_threshold=surface_contact_threshold,
            pocket_neighbor_kinds=pocket_neighbor_kinds,
        )

    def setUprDockGrid(
        self,
        grid_folder,
        center=(10,10,10),
        mol2_input=True,
        models=None,
        exclude_models=None,
    ):
        """
        Setup grid calculation for each model.

        Parameters
        ==========
        grid_folder : str
            Path to grid calculation folder
        center_atoms : tuple
            Atoms to center the grid box.
        cst_positions : dict
            atom and radius for cst position for each model:
            cst_positions = {
            model : ((chain_id, residue_index, atom_name), radius), ...
            }
        """

        # Create grid job folders
        if not os.path.exists(grid_folder):
            os.mkdir(grid_folder)

        if not os.path.exists(grid_folder + "/input_models"):
            os.mkdir(grid_folder + "/input_models")

        if not os.path.exists(grid_folder + "/output_models"):
            os.mkdir(grid_folder + "/output_models")

        for model in self.models_names:
            if not os.path.exists(grid_folder + "/output_models/" + model):
                os.mkdir(grid_folder + "/output_models/"+model)

        if isinstance(models, str):
            models = [models]

        if isinstance(exclude_models, str):
            exclude_models = [exclude_models]

        center_string = str(center)
        strip_center = center_string.replace(" ", "")

        # Save all input models
        self.saveModels(grid_folder + "/input_models", convert_to_mol2=mol2_input, models=models)

        # Create grid input files
        jobs = []
        for model in self.models_names:

            if models and model not in models:
                continue

            if exclude_models and model in exclude_models:
                continue
            # Write grid input file
            with open(grid_folder + "/output_models/"+model+"/" + model + ".prm", "w") as gif:

                gif.write("RBT_PARAMETER_FILE_V1.00 \n")
                gif.write("TITLE rdock \n")
                gif.write(" \n")
                gif.write("RECEPTOR_FILE "+"../../input_models/"+model+".mol2 \n")
                gif.write("RECEPTOR_FLEX 3.0 \n")
                gif.write(" \n")
                gif.write("##################################################################\n")
                gif.write("### CAVITY DEFINITION:\n")
                gif.write("##################################################################\n")
                gif.write("SECTION MAPPER\n")
                gif.write("    SITE_MAPPER RbtSphereSiteMapper\n")
                gif.write("    CENTER "+strip_center+" \n")
                gif.write("    RADIUS 10\n")
                gif.write("    SMALL_SPHERE 1.0\n")
                gif.write("    MIN_VOLUME 100\n")
                gif.write("    MAX_CAVITIES 1\n")
                gif.write("    VOL_INCR 0.0\n")
                gif.write("    GRIDSTEP 0.5\n")
                gif.write("END_SECTION\n")
                gif.write("\n")
                gif.write("#################################\n")
                gif.write("#CAVITY RESTRAINT PENALTY\n")
                gif.write("#################################\n")
                gif.write("SECTION CAVITY\n")
                gif.write("    SCORING_FUNCTION RbtCavityGridSF\n")
                gif.write("    WEIGHT 1.0\n")
                gif.write("END_SECTION\n")


            command = 'module load rdock \n'
            command += "cd " + grid_folder + "/output_models/"+model+"\n"

            # Add grid generation command

            command += "rbcavity -was -d -r " + model + ".prm > " + model+".log \n"
            command += "cd ../../.. \n"
            jobs.append(command)

        return jobs

    def setUprDockDocking(
        self,
        grid_folder,
        ligand_folder,
        n=100,
        models=None,
        exclude_models=None,
    ):
        """
        Setup rdock calculation for each model.
        First you need to run the grid calculation (setUprDockGrid)
        You need to upload the folder where you have your ligands in .sd format to the place you want to run the docking

        Parameters
        ==========
        grid_folder : str
            Path to grid calculation folder
        ligand_folder : str
            Path to the ligand folder where ligands in .sd format are stored.
        n : int
            Number of poses to generate
        """

        if isinstance(models, str):
            models = [models]

        if isinstance(exclude_models, str):
            exclude_models = [exclude_models]

        #Get the ligand in the ligand folder
        ### ADD to get only .sd
        folder_path = str(ligand_folder)
        ligand_files = [f for f in os.listdir(folder_path) if f.endswith(".sd")]

        # Create grid input files
        jobs = []
        for model in self.models_names:
            for ligand in ligand_files:

                command = 'module load rdock \n'
                command += "cd " + grid_folder + "/output_models/"+model+"\n"

                # Add docking command
                command += "rbdock -i ../../../" + ligand_folder+"/"+ligand+ " -o "+ model + "_results.out -r " + model+".prm " + "-p dock.prm -n " + str(n) + " -allH \n"
                command += "cd ../../.. \n"
                jobs.append(command)

        return jobs

    def analyseRdockDockings(self, docking_folder,protocol="score"):
        """
        Analyse rDock calculations. Only protocol "score" works for now
        It generates a folder with all the .csv files
        Parameters
        ==========
        docking_folder : str
            Path to rdock calculation folder
        protocol : str
            Protocol to use
        """

        import csv
        if protocol == 'dock':

            # Folder path containing the files
            folder_path = str(docking_folder)
            storage_path = str(docking_folder)+"results"
            # Create grid job folders
            if not os.path.exists(storage_path):
                os.mkdir(storage_path)

            data = []
            for model in self.models_names:

                for filename in [x for x in os.listdir(docking_folder +"/output_models/" + model) if x.endswith("_results.out.sd")]:
                    folder_path_model = docking_folder +"output_models/" + model
                    file_path = os.path.join(folder_path_model, filename)

                    counter = 1
                    score_bool = False
                    conformer_bool = False

                    # Open the file
                    with open(file_path, 'r') as file:
                        for line in file:
                            if score_bool:
                                score = line.split()[0]
                            if conformer_bool:
                                ligand, conformer = line.split('-')
                                data.append(
                                    [filename, counter, ligand, conformer, score])
                            if '$$$$' in line:
                                counter += 1
                            if '>  <SCORE>' in line:
                                score_bool = True
                            else:
                                score_bool = False
                            if '>  <s_lp_Variant>' in line:
                                conformer_bool = True
                            else:
                                conformer_bool = False
                # Write the extracted data to a CSV file
                output_file = '{}_rDock_data.csv'.format(model)
                #print(storage_path, output_file)
                with open(os.path.join(storage_path, output_file), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['file_name', 'file_entry', 'ligand',
                                 'conformer', 'rdock_score'])
                    writer.writerows(data)

                print(' - rDock data extraction completed.')
                print(' - Data saved in {}'.format(os.path.join(storage_path, output_file)))

        elif protocol == 'score':

            # Folder path containing the files
            folder_path = str(docking_folder)
            storage_path = str(docking_folder)+"results"
            # Create grid job folders
            if not os.path.exists(storage_path):
                os.mkdir(storage_path)


            for model in self.models_names:
                for filename in [x for x in os.listdir(docking_folder +"/output_models/" + model) if x.endswith("_results.out.sd")]:
                    folder_path_model = docking_folder +"output_models/" + model
                    file_path = os.path.join(folder_path_model, filename)

                    ligand = filename

                    counter = 1
                    score_bool = False
                    conformer_bool = False

                # Open the file
                    data = []
                    with open(file_path, 'r') as file:
                        for line in file:
                            if score_bool:
                                score = line.split()[0]
                                data.append(
                                [filename, counter, ligand, score])
                            if '$$$$' in line:
                                counter += 1
                            if '>  <SCORE>' in line:
                                score_bool = True
                            else:
                                score_bool = False

                # Write the extracted data to a CSV file
                output_file = '{}_rDock_data.csv'.format(model)
                with open(os.path.join(storage_path, output_file), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        ['file_name', 'file_entry', 'ligand', 'rdock_score'])
                    writer.writerows(data)

                print(' - rDock data extraction completed.')
                print(' - Data saved in {}'.format(os.path.join(storage_path, output_file)))


    def setUpRosettaDocking(self, docking_folder, ligands_pdb_folder=None, ligands_sdf_folder=None,
                            param_files=None, cst_files=None,
                            coordinates=None, smiles_file=None, sdf_file=None, docking_protocol='repack',
                            high_res_cycles=None, high_res_repack_every_Nth=None, num_conformers=50,
                            prune_rms_threshold=0.5, max_attempts=1000, rosetta_home=None, separator='-',
                            use_exp_torsion_angle_prefs=True, use_basic_knowledge=True, only_scorefile=False,
                            enforce_chirality=True, skip_conformers_if_found=False, overwrite=False, skip_finished=False, skip_silent_file=False, grid_width=10.0,
                            n_jobs=1, python2_executable='python2.7', store_initial_placement=False,
                            pdb_output=False, atom_pairs=None, angles=None, parallelisation="srun",
                            executable='rosetta_scripts.mpi.linuxgccrelease', ligand_chain='B', nstruct=100):

        """
        Set up docking calculations using Rosetta.

        Parameters:
        -----------
        docking_folder : str
            Path to the folder where docking results will be stored.
        ligands_pdb_folder : str, optional
            Path to the folder containing ligand PDB files. Mutually exclusive with `ligands_sdf_folder`, `smiles_file`, and `sdf_file`.
        ligands_sdf_folder : str, optional
            Path to the folder containing ligand SDF files. Mutually exclusive with `ligands_pdb_folder`, `smiles_file`, and `sdf_file`.
        smiles_file : str, optional
            Path to a file containing ligand SMILES strings. Mutually exclusive with `ligands_pdb_folder`, `ligands_sdf_folder`, and `sdf_file`.
        sdf_file : str, optional
            Path to an SDF file containing ligands. Mutually exclusive with `ligands_pdb_folder`, `ligands_sdf_folder`, and `smiles_file`.
        coordinates : (list, tuple, dict), optional
            The per-model coordinates dictionary to position the ligand (it can be multiple coordinates).
            If a list or tuples of coordinates is given, then they will be used for all the models.
        docking_protocol : str, optional
            Docking protocol to use. Available options are 'repack', 'mcm', and 'custom'. Default is 'repack'.
        high_res_cycles : int, optional
            Number of cycles for high-resolution docking when using the 'custom' protocol.
            Must be provided if 'custom' protocol is selected.
        high_res_repack_every_Nth : int, optional
            Repack frequency for high-resolution docking when using the 'custom' protocol.
            Must be provided if 'custom' protocol is selected.
        num_conformers : int, optional
            Number of conformers to generate for each ligand. Default is 50.
        prune_rms_threshold : float, optional
            RMSD threshold for pruning similar conformers. Default is 0.5 Å.
        max_attempts : int, optional
            Maximum number of attempts for embedding conformers. Default is 1000.
        rosetta_home : str, optional
            Optional explicit Rosetta home used only when `ROSETTA_HOME` is not already defined in the
            environment (including sourced `~/.bashrc`). If both are present, the environment variable
            takes precedence and a warning is emitted.
        separator : str, optional
            Separator character to use in file names. Default is '-'.
        use_exp_torsion_angle_prefs : bool, optional
            Use experimental torsion angle preferences in embedding. Default is True.
        use_basic_knowledge : bool, optional
            Use basic knowledge such as planarity of aromatic rings. Default is True.
        enforce_chirality : bool, optional
            Enforce correct chiral centers during embedding. Default is True.
        skip_conformers_if_found : bool, optional
            Skip generating conformers if they are already found. Default is False.
        overwrite : bool, optional
            Re-generate conformers and parameter files even if expected outputs already exist. Default is False.
        skip_finished : bool, optional
            Skip setting up model/ligand combinations when expected docking outputs already exist. Default is False.
        skip_silent_file : bool, optional
            Do not write silent (.out) files; only write score files. Default is False.
        grid_width : float, optional
            Width of the scoring grid. Default is 10.0.
        n_jobs : int, optional
            Number of parallel jobs to run. Default is 1.
        python2_executable : str, optional
            Path to the Python 2 executable. Default is 'python2.7'.
        executable : str, optional
            Path to the Rosetta scripts executable. Default is 'rosetta_scripts.mpi.linuxgccrelease'.
        ligand_chain : str, optional
            Chain identifier for the ligand. Default is 'B'.
        nstruct : int, optional
            Number of output structures to generate. Default is 100.
        atom_pairs : dict, optional
            Distance definitions matching the format accepted by :meth:`analyseRosettaDocking`:
            ``{model: {ligand: [((chain, res, atom), ligand_atom), ...]}}`` where ``ligand_atom`` may be either a tuple
            ``(chain, res, atom)`` or the ligand atom name. The function will create Rosetta ``AtomicDistance`` filters
            for each pair and will resolve ligand atom names using the generated ligand params.
        angles : dict, optional
            Currently unsupported; RosettaScripts does not provide a convenient ``atomicAngle`` mover.
        cst_files : dict, optional
            Mapping from model names to ligand dictionaries mirroring ``atom_pairs``:
                ``{model: {ligand: [constraint_file1, constraint_file2, ...]}}``.
            Each referenced file is copied to ``docking_folder/cst_files/<model>`` and loaded only for the
            matching model/ligand combination.

        Raises:
        -------
        ValueError
            If an invalid docking protocol is specified or if required parameters for the
            'custom' protocol are not provided, or if mutually exclusive inputs are given.

        Notes:
        ------
        This function prepares the necessary folders and XML script for running Rosetta docking
        simulations. It sets up score functions, ligand areas, interface builders, movemap builders,
        scoring grids, and movers based on the specified protocol. It also generates conformers for
        each ligand using RDKit and writes them to an SDF file with the PDB file name included if
        input is from PDB files, SMILES, or an SDF file.

        The generated XML script is saved in the specified docking folder.
        """

        def generate_conformers(mol, num_conformers=50, prune_rms_threshold=0.5, max_attempts=1000,
                                use_exp_torsion_angle_prefs=True, use_basic_knowledge=True,
                                enforce_chirality=True):
            """
            Generate and optimize conformers for a molecule.

            Parameters:
            -----------
            mol : rdkit.Chem.Mol
                The RDKit molecule for which to generate conformers.
            num_conformers : int, optional
                Number of conformers to generate. Default is 50.
            prune_rms_threshold : float, optional
                RMSD threshold for pruning similar conformers. Default is 0.5 Å.
            max_attempts : int, optional
                Maximum number of attempts for embedding conformers. Default is 1000.
            use_exp_torsion_angle_prefs : bool, optional
                Use experimental torsion angle preferences in embedding. Default is True.
            use_basic_knowledge : bool, optional
                Use basic knowledge such as planarity of aromatic rings. Default is True.
            enforce_chirality : bool, optional
                Enforce correct chiral centers during embedding. Default is True.

            Returns:
            --------
            rdkit.Chem.Mol
                RDKit molecule with embedded conformers.
            """
            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Ensure aromaticity is detected
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)

            # Embed multiple conformers
            params = AllChem.ETKDGv3()
            params.numThreads = 0  # Use all available threads
            params.pruneRmsThresh = prune_rms_threshold
            params.maxAttempts = max_attempts
            params.useExpTorsionAnglePrefs = use_exp_torsion_angle_prefs
            params.useBasicKnowledge = use_basic_knowledge
            params.enforceChirality = enforce_chirality

            conformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

            # Optimize each conformer using UFF
            for conf_id in conformers:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

            return mol

        def write_molecule_to_sdf(mol, output_sdf_path):
            """
            Write the molecule with conformers to an SDF file with explicit aromatic bonds.

            Parameters:
            -----------
            mol : rdkit.Chem.Mol
                The RDKit molecule with conformers.
            output_sdf_path : str
                Path to the output SDF file.
            """
            sdf_writer = Chem.SDWriter(output_sdf_path)
            sdf_writer.SetKekulize(False)  # Ensure aromatic bonds are not Kekulized
            for conf_id in range(mol.GetNumConformers()):
                sdf_writer.write(mol, confId=conf_id)
            sdf_writer.close()

        def process_ligand_file(ligand_file_path, base_name, ligand_dir):
            """
            Process a single ligand file (PDB, SDF, or SMILES) to generate conformers and write to SDF.

            Parameters:
            -----------
            ligand_file_path : str
                Path to the ligand file.
            base_name : str
                Base name of the ligand.
            ligand_dir : str
                Folder to store the output SDF file (the ligand's params directory).
            """
            os.makedirs(ligand_dir, exist_ok=True)
            output_sdf_path = os.path.join(ligand_dir, f"{base_name}.sdf")

            if (skip_conformers_if_found or not overwrite) and os.path.exists(output_sdf_path):
                return base_name

            # Determine file type and load molecule
            if ligand_file_path.endswith('.pdb'):
                mol = Chem.MolFromPDBFile(ligand_file_path, removeHs=False)
            elif ligand_file_path.endswith('.sdf'):
                mol_supplier = Chem.SDMolSupplier(ligand_file_path, removeHs=False)
                mol = mol_supplier[0]
            else:
                raise ValueError(f"Unsupported file type: {ligand_file_path}")

            if mol is None:
                raise ValueError(f"Could not read file: {ligand_file_path}")

            # Generate conformers
            mol_with_conformers = generate_conformers(mol, num_conformers=num_conformers,
                                                      prune_rms_threshold=prune_rms_threshold,
                                                      max_attempts=max_attempts,
                                                      use_exp_torsion_angle_prefs=use_exp_torsion_angle_prefs,
                                                      use_basic_knowledge=use_basic_knowledge,
                                                      enforce_chirality=enforce_chirality)

            # Set the molecule name
            mol_with_conformers.SetProp("_Name", base_name)

            # Write to SDF file
            write_molecule_to_sdf(mol_with_conformers, output_sdf_path)
            return base_name

        def make_param_file(input_sdf_path, ligand_name, output_dir):
            """
            Generate a parameter file for a ligand.

            Parameters:
            -----------
            input_sdf_path : str
                Path to the input SDF file containing conformers.
            ligand_name : str
                Name of the ligand.
            output_dir : str
                Directory to save the generated parameter file.

            Returns:
            --------
            str
                Path to the generated parameter file.
            """
            full_dir = os.path.join(output_dir, ligand_name)
            os.makedirs(full_dir, exist_ok=True)

            output_param_path = f"{ligand_name}.params"
            output_pdb_path = f"{ligand_name}.pdb"
            output_conformer_path = f"{ligand_name}_conformers.pdb"

            existing_param = os.path.join(full_dir, output_param_path)
            existing_pdb = os.path.join(full_dir, output_pdb_path)
            existing_conformers = os.path.join(full_dir, output_conformer_path)
            expected_files_present = (
                os.path.exists(existing_param)
                and os.path.exists(existing_pdb)
                and os.path.exists(existing_conformers)
                and os.path.exists(input_sdf_path)
            )
            if not overwrite and expected_files_present:
                return existing_param

            param_file_command = (
                f"{rosetta_home}/main/source/scripts/python/public/molfile_to_params.py -n {ligand_name} "
                f"--long-names --clobber --conformers-in-one-file --mm-as-virt {input_sdf_path}"
            )

            subprocess.call(param_file_command, shell=True)

            missing_files: List[str] = []
            cwd = os.getcwd()

            try:
                shutil.move(os.path.join(cwd, output_param_path), os.path.join(full_dir, output_param_path))
            except IOError:
                missing_files.append(output_param_path)

            try:
                shutil.move(os.path.join(cwd, output_pdb_path), os.path.join(full_dir, output_pdb_path))
            except IOError:
                missing_files.append(output_pdb_path)

            try:
                shutil.move(os.path.join(cwd, output_conformer_path), os.path.join(full_dir, output_conformer_path))
            except IOError:
                missing_files.append(output_conformer_path)

            # If conformers file exists but is empty, drop the PDB_ROTAMERS reference to avoid Rosetta errors
            conformer_full_path = os.path.join(full_dir, output_conformer_path)
            params_full_path = os.path.join(full_dir, output_param_path)
            if os.path.exists(conformer_full_path) and os.path.getsize(conformer_full_path) == 0:
                if os.path.exists(params_full_path):
                    with open(params_full_path, "r") as pf:
                        lines = pf.readlines()
                    with open(params_full_path, "w") as pf:
                        for line in lines:
                            if line.strip().startswith("PDB_ROTAMERS"):
                                continue
                            pf.write(line)
                    warnings.warn(
                        f"Removed empty conformer reference from params for ligand '{ligand_name}' "
                        f"({os.path.relpath(conformer_full_path, full_dir)} is empty).",
                        UserWarning,
                    )
                else:
                    warnings.warn(
                        f"Conformer file for ligand '{ligand_name}' is empty but params file was not found.",
                        UserWarning,
                    )

            sdf_name = os.path.basename(input_sdf_path)
            if not os.path.exists(os.path.join(full_dir, sdf_name)):
                missing_files.append(sdf_name)

            if missing_files:
                missing_list = ", ".join(sorted(set(missing_files)))
                warnings.warn(
                    f"molfile_to_params did not produce expected files for ligand '{ligand_name}' in {full_dir}: {missing_list}",
                    UserWarning,
                )

            if output_param_path in missing_files:
                return None

            return os.path.join(full_dir, output_param_path)

        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as e:
            raise ImportError("RDKit is not installed. Please install it to use the setUpRosettaDocking function.")

        def _load_rosetta_from_bashrc():
            bashrc = os.path.expanduser("~/.bashrc")
            if not os.path.exists(bashrc):
                return None
            result = subprocess.run(
                ["bash", "-lc", f"source {bashrc} >/dev/null 2>&1 && printf '%s' \"$ROSETTA_HOME\""],
                capture_output=True,
                text=True,
                check=False,
            )
            value = result.stdout.strip()
            if value:
                os.environ["ROSETTA_HOME"] = value
                return value
            return None

        env_rosetta_home = os.environ.get("ROSETTA_HOME")
        if not env_rosetta_home:
            env_rosetta_home = _load_rosetta_from_bashrc()

        final_rosetta_home = env_rosetta_home
        if env_rosetta_home and rosetta_home and os.path.abspath(rosetta_home) != os.path.abspath(env_rosetta_home):
            warnings.warn(
                "ROSETTA_HOME environment variable is already set and will be used; the `rosetta_home` argument is ignored.",
                UserWarning,
            )
        elif not env_rosetta_home and rosetta_home:
            final_rosetta_home = rosetta_home

        if not final_rosetta_home:
            raise ValueError("The ROSETTA_HOME environment variable is not set; please export it before calling setUpRosettaDocking.")

        os.environ["ROSETTA_HOME"] = final_rosetta_home

        rosetta_home = final_rosetta_home

        if angles:
            raise ValueError('Angles has not being implemented. Rosetta scripts do not have an easy function to compute angles. Perhaps with CreateAngleConstraint mover?')

        # Check for mutually exclusive inputs
        if sum([bool(ligands_pdb_folder), bool(ligands_sdf_folder), bool(smiles_file), bool(sdf_file)]) != 1:
            raise ValueError('Specify exactly one of ligands_pdb_folder, ligands_sdf_folder, smiles_file, or sdf_file.')

        # Check given separator compatibility
        for model in self:
            if separator in model:
                raise ValueError(f'The given separator {separator} was found in model {model}. Please use a different one.')

        # Warn if coordinates were not provided
        if coordinates is None:
            warnings.warn(
                "Rosetta docking routines require explicit ligand placement coordinates; "
                "provide the `coordinates` argument so setUpRosettaDocking can construct the XML movers.",
                UserWarning,
            )

        # Uniform coordinates format
        if isinstance(coordinates, (list, tuple)):
            if isinstance(coordinates[0], (list, tuple)) and len(coordinates[0]) == 3:
                tmp = {model: coordinates for model in self}
                coordinates = tmp
            elif len(coordinates) == 3 and isinstance(coordinates[0], (int, float)):
                tmp = {model: [coordinates] for model in self}
                coordinates = tmp
            else:
                raise ValueError('Check your given coordinates format')
        elif not isinstance(coordinates, dict):
            raise ValueError('Check your given coordinates format')

        # Check given protocol
        available_protocols = ['repack', 'mcm', 'custom']
        if docking_protocol not in available_protocols:
            raise ValueError(f'Invalid protocol. Available protocols are: {available_protocols}')

        if docking_protocol == 'custom' and (not high_res_cycles or not high_res_repack_every_Nth):
            raise ValueError('You must provide high_res_cycles and high_res_repack_every_Nth for the custom protocol.')

        # Create docking job folders
        os.makedirs(docking_folder, exist_ok=True)
        xml_folder = os.path.join(docking_folder, 'xml')
        ligand_params_folder = os.path.join(docking_folder, 'ligand_params')
        input_models_folder = os.path.join(docking_folder, 'input_models')
        flags_folder = os.path.join(docking_folder, 'flags')
        output_folder = os.path.join(docking_folder, 'output_models')

        for folder in [xml_folder, ligand_params_folder, input_models_folder,
                       flags_folder, output_folder]:
            os.makedirs(folder, exist_ok=True)

        # Write model structures
        self.saveModels(input_models_folder)

        # Create score functions
        ligand_soft_rep = rosettaScripts.scorefunctions.new_scorefunction('ligand_soft_rep',
                                                                          weights_file='ligand_soft_rep')
        ligand_soft_rep.addReweight('fa_elec', weight=0.42)
        ligand_soft_rep.addReweight('hbond_bb_sc', weight=1.3)
        ligand_soft_rep.addReweight('hbond_sc', weight=1.3)
        ligand_soft_rep.addReweight('rama', weight=0.2)

        ligand_hard_rep = rosettaScripts.scorefunctions.new_scorefunction('ligand_hard_rep',
                                                                          weights_file='ligand')
        ligand_hard_rep.addReweight('fa_intra_rep', weight=0.004)
        ligand_hard_rep.addReweight('fa_elec', weight=0.42)
        ligand_hard_rep.addReweight('hbond_bb_sc', weight=1.3)
        ligand_hard_rep.addReweight('hbond_sc', weight=1.3)
        ligand_hard_rep.addReweight('rama', weight=0.2)

        # Clone the hard rep scorefunction but silence constraint terms so ligand metrics
        # can be computed without bias from ligand constraints.
        ligand_hard_rep_no_cst = rosettaScripts.scorefunctions.new_scorefunction(
            'ligand_hard_rep_no_cst', weights_file='ligand'
        )
        ligand_hard_rep_no_cst.addReweight('fa_intra_rep', weight=0.004)
        ligand_hard_rep_no_cst.addReweight('fa_elec', weight=0.42)
        ligand_hard_rep_no_cst.addReweight('hbond_bb_sc', weight=1.3)
        ligand_hard_rep_no_cst.addReweight('hbond_sc', weight=1.3)
        ligand_hard_rep_no_cst.addReweight('rama', weight=0.2)
        for constraint_term in (
            'atom_pair_constraint',
            'coordinate_constraint',
            'angle_constraint',
            'dihedral_constraint',
            'res_type_constraint',
            'metalbinding_constraint',
        ):
            ligand_hard_rep_no_cst.addReweight(constraint_term, weight=0.0)

        # Create ligand areas
        docking_sidechain = rosettaScripts.ligandArea('docking_sidechain',
                                                       chain=ligand_chain,
                                                       cutoff=6.0,
                                                       add_nbr_radius=True,
                                                       all_atom_mode=True,
                                                       minimize_ligand=10)

        final_sidechain = rosettaScripts.ligandArea('final_sidechain',
                                                     chain=ligand_chain,
                                                     cutoff=6.0,
                                                     add_nbr_radius=True,
                                                     all_atom_mode=True)

        final_backbone = rosettaScripts.ligandArea('final_backbone',
                                                    chain=ligand_chain,
                                                    cutoff=7.0,
                                                    add_nbr_radius=True,
                                                    all_atom_mode=True,
                                                    calpha_restraints=0.3)

        # Set interface builders
        side_chain_for_docking = rosettaScripts.interfaceBuilder('side_chain_for_docking',
                                                                 ligand_areas=docking_sidechain)
        side_chain_for_final = rosettaScripts.interfaceBuilder('side_chain_for_final',
                                                               ligand_areas=final_sidechain)
        backbone = rosettaScripts.interfaceBuilder('final_backbone',
                                                   ligand_areas=final_backbone,
                                                   extension_window=3)

        # Set movemap builders
        docking = rosettaScripts.movemapBuilder('docking', sc_interface=side_chain_for_docking)
        final = rosettaScripts.movemapBuilder('final', sc_interface=side_chain_for_final,
                                              bb_interface=backbone, minimize_water=True)

        # Set up scoring grid
        vdw = rosettaScripts.scoringGrid.classicGrid('vdw', weight=1.0)

        ### Create docking movers

        # Write coordinates
        xyz = []
        for model in coordinates:
            for coordinate in coordinates[model]:
                model_xyz = {}
                model_xyz['file_name'] = '../../input_models/'+model+'.pdb'
                model_xyz['x'] = coordinate[0]
                model_xyz['y'] = coordinate[1]
                model_xyz['z'] = coordinate[2]
                xyz.append(model_xyz)
        with open(docking_folder+'/coordinates.json', 'w') as jf:
            json.dump(xyz, jf)

        # Create start from mover for initial ligand positioning
        startFrom = rosettaScripts.movers.startFrom(chain=ligand_chain)
        startFrom.addFile('../../coordinates.json')

        if store_initial_placement:
            store_initial_placement = rosettaScripts.movers.dumpPdb(name='store_initial_placement', file_name='initial_placement.pdb')

        # Create transform mover
        transform = rosettaScripts.movers.transform(chain=ligand_chain, box_size=5.0, move_distance=0.1, angle=5,
                                                    cycles=500, repeats=1, temperature=5, initial_perturb=5.0)

        # Create high-resolution docking mover according to the employed docking protocol
        if docking_protocol == 'repack':
            cycles = 1
            repack_every_Nth = 1
        elif docking_protocol == 'mcm':
            cycles = 6
            repack_every_Nth = 3
        elif docking_protocol == 'custom':
            cycles = high_res_cycles
            repack_every_Nth = high_res_repack_every_Nth

        highResDocker = rosettaScripts.movers.highResDocker(cycles=cycles,
                                                            repack_every_Nth=repack_every_Nth,
                                                            scorefxn=ligand_soft_rep,
                                                            movemap_builder=docking)

        # Create final minimization mover
        finalMinimizer = rosettaScripts.movers.finalMinimizer(scorefxn=ligand_hard_rep, movemap_builder=final)

        ### Create combined protocols

        # Create low-resolution mover
        low_res_dock_movers = [transform]
        low_res_dock = rosettaScripts.movers.parsedProtocol('low_res_dock', low_res_dock_movers)

        # Create high-resolution mover
        high_res_dock_movers = [highResDocker, finalMinimizer]
        high_res_dock = rosettaScripts.movers.parsedProtocol('high_res_dock', high_res_dock_movers)

        # Convert ligand input to conformers
        ligands = []

        # Handle PDB ligand files
        if ligands_pdb_folder:
            for ligand_file in os.listdir(ligands_pdb_folder):
                if not ligand_file.endswith('.pdb'):
                    continue
                ligand_file_path = os.path.join(ligands_pdb_folder, ligand_file)
                base_name = os.path.splitext(ligand_file)[0]
                ligand_dir = os.path.join(ligand_params_folder, base_name)
                ligands.append(process_ligand_file(ligand_file_path, base_name, ligand_dir))

        # Handle SDF ligand files
        elif ligands_sdf_folder:
            for ligand_file in os.listdir(ligands_sdf_folder):
                if not ligand_file.endswith('.sdf'):
                    continue
                ligand_file_path = os.path.join(ligands_sdf_folder, ligand_file)
                base_name = os.path.splitext(ligand_file)[0]
                ligand_dir = os.path.join(ligand_params_folder, base_name)
                ligands.append(process_ligand_file(ligand_file_path, base_name, ligand_dir))

        # Handle SMILES input
        elif smiles_file:
            with open(smiles_file, 'r') as f:
                for line in f:
                    smi, name = line.strip().split()
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        raise ValueError(f"Could not parse SMILES: {smi}")
                    mol = Chem.AddHs(mol)
                    mol.SetProp("_Name", name)
                    ligand_dir = os.path.join(ligand_params_folder, name)
                    os.makedirs(ligand_dir, exist_ok=True)
                    output_sdf_path = os.path.join(ligand_dir, f"{name}.sdf")

                    if (skip_conformers_if_found or not overwrite) and os.path.exists(output_sdf_path):
                        ligands.append(name)
                        continue

                    ligands.append(name)
                    mol_with_conformers = generate_conformers(mol, num_conformers=num_conformers,
                                                              prune_rms_threshold=prune_rms_threshold,
                                                              max_attempts=max_attempts,
                                                              use_exp_torsion_angle_prefs=use_exp_torsion_angle_prefs,
                                                              use_basic_knowledge=use_basic_knowledge,
                                                              enforce_chirality=enforce_chirality)
                    write_molecule_to_sdf(mol_with_conformers, output_sdf_path)

        # Handle SDF input
        elif sdf_file:
            suppl = Chem.SDMolSupplier(sdf_file)
            for mol in suppl:
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)
                name = mol.GetProp("_Name")
                ligand_dir = os.path.join(ligand_params_folder, name)
                os.makedirs(ligand_dir, exist_ok=True)
                output_sdf_path = os.path.join(ligand_dir, f"{name}.sdf")

                if (skip_conformers_if_found or not overwrite) and os.path.exists(output_sdf_path):
                    ligands.append(name)
                    continue

                ligands.append(name)
                mol_with_conformers = generate_conformers(mol, num_conformers=num_conformers,
                                                          prune_rms_threshold=prune_rms_threshold,
                                                          max_attempts=max_attempts,
                                                          use_exp_torsion_angle_prefs=use_exp_torsion_angle_prefs,
                                                          use_basic_knowledge=use_basic_knowledge,
                                                          enforce_chirality=enforce_chirality)
                write_molecule_to_sdf(mol_with_conformers, output_sdf_path)

        model_ligand_residue_index: Dict[str, int] = {}
        for model in self:
            residue_count = 0
            for residue in self.structures[model].get_residues():
                if residue.get_resname() == "HOH":
                    continue
                residue_count += 1
            if residue_count == 0:
                raise ValueError(f"Model '{model}' has no residues to anchor the ligand.")
            ligand_residue_index = residue_count + 1
            model_ligand_residue_index[model] = ligand_residue_index

        requested_atom_pairs: Dict[
            Tuple[str, str],
            List[Tuple[Tuple[str, int, str], Union[str, Tuple[str, int, str]]]],
        ] = {}
        ligand_atom_cache: Dict[str, Dict[str, Tuple[str, int, str]]] = {}
        if atom_pairs is not None:
            if not isinstance(atom_pairs, dict):
                raise ValueError(
                    "atom_pairs must be a dictionary structured as {model: {ligand: [(protein_atom, ligand_atom), ...]}}."
                )

            available_models = set(self.structures.keys())
            available_ligands = set(ligands)

            for model_name, ligand_dict in atom_pairs.items():
                if model_name not in available_models:
                    raise ValueError(
                        f"Model '{model_name}' from atom_pairs is not part of the docking setup."
                    )
                if not isinstance(ligand_dict, dict):
                    raise ValueError(f"atom_pairs[{model_name!r}] must map ligands to atom-pair lists.")

                for ligand_name, pair_entries in ligand_dict.items():
                    if ligand_name not in available_ligands:
                        raise ValueError(
                            f"Ligand '{ligand_name}' from atom_pairs is not part of the prepared ligand set."
                        )
                    if not pair_entries:
                        continue
                    normalized_pairs: List[
                        Tuple[Tuple[str, int, str], Union[str, Tuple[str, int, str]]]
                    ] = []
                    for entry in pair_entries:
                        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                            raise ValueError(
                                "Each atom_pairs entry must be a 2-tuple: ((chain, residue, atom), ligand_atom)."
                            )
                        protein_atom, ligand_atom = entry
                        if not (isinstance(protein_atom, (list, tuple)) and len(protein_atom) == 3):
                            raise ValueError(
                                "Protein atoms in atom_pairs must be (chain, residue, atom) tuples."
                            )
                        chain_id, resseq, atom_name = protein_atom
                        try:
                            resseq_int = int(resseq)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"Residue index '{resseq}' in atom_pairs for model '{model_name}' "
                                f"and ligand '{ligand_name}' is not an integer."
                            ) from exc
                        normalized_pairs.append(((str(chain_id), resseq_int, str(atom_name)), ligand_atom))
                    if normalized_pairs:
                        requested_atom_pairs[(model_name, ligand_name)] = normalized_pairs

        def _resolve_ligand_atom_tuple(ligand_name: str, atom_identifier, override=None):
            if isinstance(atom_identifier, str):
                mapping = ligand_atom_cache.get(ligand_name)
                if mapping is None:
                    mapping = _parse_ligand_atom_map(docking_folder, ligand_name)
                    ligand_atom_cache[ligand_name] = mapping
                if atom_identifier not in mapping:
                    available = ", ".join(sorted(mapping.keys())[:10])
                    raise ValueError(
                        f"Atom '{atom_identifier}' not found in ligand '{ligand_name}'. "
                        f"Available atoms: {available}{'...' if len(mapping) > 10 else ''}"
                    )
                chain_id, resseq, atom_name = mapping[atom_identifier]
                if override is not None:
                    chain_id, resseq = override
                return (str(chain_id), int(resseq), str(atom_name))
            if isinstance(atom_identifier, (list, tuple)) and len(atom_identifier) == 3:
                chain_id, resseq, atom_name = atom_identifier
                try:
                    resseq_int = int(resseq)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Residue index '{resseq}' in ligand atom tuple for ligand '{ligand_name}' is not an integer."
                    ) from exc
                return (str(chain_id), resseq_int, str(atom_name))
            raise ValueError(
                "Ligand atoms in atom_pairs must be either an atom name string or a (chain, residue, atom) tuple."
            )

        has_external_params = False
        params_folder = os.path.join(docking_folder, "params")

        # Resolve external params by mirroring them into a dedicated params folder
        if param_files is not None:

            if isinstance(param_files, str):
                param_files = [param_files]

            resolved_params: List[str] = []
            for param in param_files:
                if os.path.isdir(param):
                    for entry in os.listdir(param):
                        if entry.endswith(".params"):
                            resolved_params.append(os.path.join(param, entry))
                else:
                    resolved_params.append(param)

            os.makedirs(params_folder, exist_ok=True)

            for param in resolved_params:
                abs_param = os.path.abspath(param)
                if not os.path.exists(abs_param):
                    raise FileNotFoundError(f"Parameter file not found: {abs_param}")

                destination = os.path.join(params_folder, os.path.basename(abs_param))
                if os.path.abspath(abs_param) != os.path.abspath(destination):
                    shutil.copyfile(abs_param, destination)
                has_external_params = True

                # Leave any auxiliary PDB/conformer files in their original locations to
                # keep user-supplied parameter sets separate from the ligand params
                # generated by this function.

        # Process each ligand
        jobs = []
        for ligand in ligands:

            # Add chain mover
            ligand_pdb = '../../ligand_params/'+ligand+'/'+ligand+'.pdb'
            addLigand = rosettaScripts.movers.addChain(name='addLigand', update_PDBInfo=True, file_name=ligand_pdb)

            # make params
            input_sdf_path = os.path.join(ligand_params_folder, ligand, f"{ligand}.sdf")
            output_dir = ligand_params_folder
            make_param_file(input_sdf_path, ligand, output_dir)

            # Process each model
            for model in self:

                # Determine expected output filenames for this model/ligand pair
                expected_score_file = f'{model}{separator}{ligand}.sc'
                expected_silent_file = None
                if not only_scorefile and not pdb_output and not skip_silent_file:
                    expected_silent_file = f'{model}{separator}{ligand}.out'

                model_output_folder = os.path.join(output_folder, model+separator+ligand)

                if skip_finished:
                    expected_score_path = os.path.join(model_output_folder, expected_score_file)
                    expected_silent_path = os.path.join(model_output_folder, expected_silent_file) if expected_silent_file else None

                    score_ok = os.path.exists(expected_score_path)
                    silent_ok = True if skip_silent_file else ((expected_silent_file is None) or (expected_silent_path and os.path.exists(expected_silent_path)))
                    pdb_ok = True
                    if pdb_output and os.path.isdir(model_output_folder):
                        pdb_ok = any(f.lower().endswith(".pdb") for f in os.listdir(model_output_folder))

                    if score_ok and silent_ok and pdb_ok:
                        continue

                # Create xml ligand docking
                xml = rosettaScripts.xmlScript()

                # Add score functions
                xml.addScorefunction(ligand_soft_rep)
                xml.addScorefunction(ligand_hard_rep)
                xml.addScorefunction(ligand_hard_rep_no_cst)

                # Add ligand areas
                xml.addLigandArea(docking_sidechain)
                xml.addLigandArea(final_sidechain)
                xml.addLigandArea(final_backbone)

                # Add interface builders
                xml.addInterfaceBuilder(side_chain_for_docking)
                xml.addInterfaceBuilder(side_chain_for_final)
                xml.addInterfaceBuilder(backbone)

                # Add movemap builders
                xml.addMovemapBuilder(docking)
                xml.addMovemapBuilder(final)

                # Prepare constraint movers list
                constraint_movers = []
                ligand_constraints_applied = False

                # Add constraint movers
                if cst_files is not None:
                    if model not in cst_files:
                        raise ValueError(f"Model {model} is not in the cst_files dictionary!")

                    model_csts = cst_files[model]
                    if isinstance(model_csts, dict):
                        if ligand not in model_csts:
                            raise ValueError(
                                f"Ligand {ligand} is not in the cst_files dictionary for model {model}."
                            )
                        ligand_cst_files = model_csts[ligand]
                    else:
                        warnings.warn(
                            "cst_files should map each model to a ligand dictionary; "
                            "the provided files will be applied to every ligand.",
                            UserWarning,
                        )
                        ligand_cst_files = model_csts

                    if isinstance(ligand_cst_files, str):
                        ligand_cst_files = [ligand_cst_files]

                    if ligand_cst_files:
                        cst_root = os.path.join(docking_folder, "cst_files")
                        os.makedirs(cst_root, exist_ok=True)
                        model_cst_dir = os.path.join(cst_root, model)
                        os.makedirs(model_cst_dir, exist_ok=True)
                        ligand_constraints_applied = True

                        for cst_file in ligand_cst_files:
                            cst_name = os.path.basename(cst_file)
                            dest_file = os.path.join(model_cst_dir, cst_name)
                            if not os.path.exists(dest_file):
                                shutil.copyfile(cst_file, dest_file)

                            set_cst = rosettaScripts.movers.constraintSetMover(
                                add_constraints=True,
                                cst_file=f"../../cst_files/{model}/{cst_name}",
                            )
                            xml.addMover(set_cst)
                            constraint_movers.append(set_cst)

                # Add scoring grid
                xml.addScoringGrid(vdw, ligand_chain=ligand_chain, width=grid_width)

                # Add doking movers
                xml.addMover(addLigand)
                xml.addMover(startFrom)
                if store_initial_placement:
                    xml.addMover(store_initial_placement)
                xml.addMover(transform)
                xml.addMover(highResDocker)
                xml.addMover(finalMinimizer)

                # Select a constraint-free score function for reporting when ligand
                # constraints were applied so interface metrics ignore penalty terms.
                if ligand_constraints_applied:
                    interface_scorefxn = ligand_hard_rep_no_cst
                    interface_score_mover_name = 'interfaceScoreCalculator_no_cst'
                else:
                    interface_scorefxn = ligand_hard_rep
                    interface_score_mover_name = 'interfaceScoreCalculator'

                interfaceScoreCalculator = rosettaScripts.movers.interfaceScoreCalculator(
                    name=interface_score_mover_name,
                    chains=ligand_chain,
                    scorefxn=interface_scorefxn,
                    compute_grid_scores=False,
                )
                reporting = rosettaScripts.movers.parsedProtocol(
                    'reporting', [interfaceScoreCalculator]
                )

                # Add scoring movers
                xml.addMover(interfaceScoreCalculator)

                # Add compund movers
                xml.addMover(low_res_dock)
                xml.addMover(high_res_dock)
                xml.addMover(reporting)

                # Set up protocol
                protocol = []
                protocol.append(addLigand)
                protocol.extend(constraint_movers)
                protocol.append(startFrom)
                if store_initial_placement:
                    protocol.append(store_initial_placement)
                protocol.append(low_res_dock)
                protocol.append(high_res_dock)
                protocol.append(reporting)

                requested_pairs = requested_atom_pairs.get((model, ligand))
                if requested_pairs:
                    normalized_pairs = []
                    for protein_atom, ligand_atom in requested_pairs:
                        override = (ligand_chain, model_ligand_residue_index[model])
                        ligand_tuple = _resolve_ligand_atom_tuple(ligand, ligand_atom, override=override)
                        normalized_pairs.append((protein_atom, ligand_tuple))

                    for protein_atom, ligand_tuple in normalized_pairs:
                        label = "distance_"
                        label += "_".join([str(x) for x in protein_atom]) + "-"
                        label += "_".join([str(x) for x in ligand_tuple])

                        d = rosettaScripts.filters.atomicDistance(
                            name=label,
                            residue1=f"{protein_atom[1]}{protein_atom[0]}",
                            atomname1=protein_atom[2],
                            residue2=f"{ligand_tuple[1]}{ligand_tuple[0]}",
                            atomname2=ligand_tuple[2],
                            distance=5.0,
                            confidence=0.0,
                        )
                        xml.addFilter(d)
                        protocol.append(d)

    #             # Add angle filters
    #             if angles:
    #                 for atoms in angles[model][ligand]:
    #                     label = "angle_"
    #                     label += "_".join([str(x) for x in atoms[0]]) + "-"
    #                     label += "_".join([str(x) for x in atoms[1]]) + "-"
    #                     label += "_".join([str(x) for x in atoms[2]])
    #                     a = rosettaScripts.filters.atomicAngle(name=label, # There is no atomicAngle function in rosetta scripts
    #                         residue1=atoms[0][0]+str(atoms[0][1]), atomname1=atoms[0][2],
    #                         residue2=atoms[1][0]+str(atoms[1][1]), atomname2=atoms[1][2],
    #                         residue3=atoms[2][0]+str(atoms[2][1]), atomname3=atoms[2][2],
    #                         angle=120.0, confidence=0.0)
    #                     xml.addFilter(a)
    #                     protocol.append(a)

                # Set protocol
                xml.setProtocol(protocol)

                # Write XML protocol file
                xml_output = os.path.join(xml_folder, f'{docking_protocol}{separator}{ligand}{separator}{model}.xml')
                xml.write_xml(xml_output)

                # Create flags files
                flags = rosettaScripts.options.flags(f'../../xml/{docking_protocol}{separator}{ligand}{separator}{model}.xml',
                                                     nstruct=nstruct, s='../../input_models/'+model+'.pdb',
                                                     output_silent_file=expected_silent_file,
                                                     output_score_file=expected_score_file)
                if only_scorefile or skip_silent_file:
                    flags.addOption('out:file:score_only')

                flags.add_ligand_docking_options()
                if has_external_params:
                    flags.addOption("in:file:extra_res_path", "../../params")
                ligand_params_file = os.path.join(ligand_params_folder, ligand, f"{ligand}.params")
                if os.path.exists(ligand_params_file):
                    flags.addOption(
                        "in:file:extra_res_fa",
                        f"../../ligand_params/{ligand}/{ligand}.params",
                        append=True,
                    )
                flags_output = os.path.join(flags_folder, f'{docking_protocol}{separator}{ligand}{separator}{model}.flags')
                flags.write_flags(flags_output)

                # Create output folder and execute commands
                os.makedirs(model_output_folder, exist_ok=True)
                command =  'cd '+model_output_folder+'\n'
                if parallelisation:
                    command += parallelisation+' '
                command += executable+' '
                command += '@ ../../flags/'+f'{docking_protocol}{separator}{ligand}{separator}{model}.flags '
                command += '\n'
                command += 'cd ../../..\n'
                jobs.append(command)

        return jobs

    def generatePointsAroundAtoms(self, atom_tuples, radii, num_points_per_radius, threshold_distance, verbose=False):
        """
        Generates equidistant points around atoms on the surfaces of spheres with different radii using the Fibonacci lattice method.
        Only stores points that are farther than a threshold distance from any heavy atom.

        :param atom_tuples: Dictionary with model names as keys and atom tuples (chain_id, residue_id, atom_name) as values
        :param radii: List of radii of the spheres
        :param num_points_per_radius: Number of points to generate on each sphere
        :param threshold_distance: Minimum distance from any heavy atom to store a point
        :param verbose: If True, print detailed debug information
        :return: Dictionary of accumulated points around atoms by model
        """
        all_model_points = {}

        for model, atom_tuple in atom_tuples.items():
            structure = self.structures[model]
            chain_id, residue_id, atom_name = atom_tuple
            target_atom = None
            heavy_atom_coords = []

            # Find the target atom and collect all heavy atom coordinates
            for chain in structure.get_chains():
                if chain.id == chain_id:
                    for residue in chain.get_residues():
                        for atom in residue.get_atoms():
                            if atom.element != 'H':  # Exclude hydrogen atoms
                                heavy_atom_coords.append(atom.coord)
                            if residue.id[1] == residue_id and atom.name == atom_name:
                                target_atom = atom

            if not target_atom:
                raise ValueError(f"Target atom not found in the structure for model {model}")

            target_atom_coord = target_atom.coord
            all_points = []

            # Convert heavy atom coordinates to numpy array for distance calculations
            heavy_atom_coords = np.array(heavy_atom_coords)
            if verbose:
                print(f"Model: {model}, Heavy atom coordinates shape: {heavy_atom_coords.shape}")

            for radius in radii:
                points = []
                # Generate equidistant points on the surface of the sphere using Fibonacci lattice
                indices = np.arange(0, num_points_per_radius, dtype=float) + 0.5
                phi = np.arccos(1 - 2 * indices / num_points_per_radius)
                theta = np.pi * (1 + 5**0.5) * indices

                if verbose:
                    print(f"Model: {model}, Generated points before filtering for radius {radius}:")
                for i in range(num_points_per_radius):
                    x = radius * np.sin(phi[i]) * np.cos(theta[i])
                    y = radius * np.sin(phi[i]) * np.sin(theta[i])
                    z = radius * np.cos(phi[i])
                    point = target_atom_coord + np.array([x, y, z])
                    if verbose:
                        print(point)

                    # Check distance to all heavy atom points
                    if heavy_atom_coords.size > 0:
                        distances = cdist([point], heavy_atom_coords)
                        if verbose:
                            print(f"Model: {model}, Distances for point {i} at radius {radius}: {distances}")
                        if np.all(distances > threshold_distance):
                            points.append(point)
                    else:
                        points.append(point)

                if verbose:
                    print(f"Model: {model}, Points after filtering for radius {radius}: {points}")
                all_points.extend(points)

            if verbose:
                print(f"Model: {model}, Accumulated points: {all_points}")
            all_model_points[model] = all_points

        return all_model_points

    def visualizePointsOnStructure(self, points_by_model, output_folder, chain_id='A', residue_name="SPH", verbose=False):
        def addResidueToStructure(structure, chain_id, resname, atom_names, coordinates, new_resid=None, elements=None, hetatom=True, water=False):
            """
            Add a new residue with given atoms to the structure.
            """
            model = structure[0]
            chain = next((chain for chain in model.get_chains() if chain_id == chain.id), None)
            if not chain:
                if verbose:
                    print(f"Chain ID {chain_id} not found. Creating a new chain.")
                chain = Chain.Chain(chain_id)
                model.add(chain)

            if coordinates.ndim == 1:
                coordinates = np.array([coordinates])

            if coordinates.shape[1] != 3 or len(coordinates) != len(atom_names):
                if verbose:
                    print(f"Atom names: {atom_names}")
                    print(f"Coordinates: {coordinates}")
                raise ValueError("Mismatch between atom names and coordinates.")

            new_resid = new_resid or max((r.id[1] for r in chain.get_residues()), default=0) + 1
            rt_flag = 'H' if hetatom else 'W' if water else ' '
            residue = PDB.Residue.Residue((rt_flag, new_resid, ' '), resname, ' ')

            serial_number = max((a.serial_number for a in chain.get_atoms()), default=0) + 1
            for i, atnm in enumerate(atom_names):
                atom = PDB.Atom.Atom(atnm, coordinates[i], 0, 1.0, " ", f"{atnm: <4}", serial_number + i, elements[i] if elements else "H")
                residue.add(atom)
            chain.add(residue)
            return new_resid

        def addPointsToStructure(structure, points, chain_id, residue_name="SPH"):
            """
            Add points as a new residue to the structure.
            """
            atom_names = [f"SP{i+1}" for i in range(len(points))]
            coordinates = np.array(points)
            elements = ['H'] * len(points)
            if verbose:
                print(f"Number of points: {len(points)}")
                print(f"Number of atom names: {len(atom_names)}")
            addResidueToStructure(structure, chain_id, residue_name, atom_names, coordinates, elements=elements, hetatom=True)
            return structure

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for model, points in points_by_model.items():
            structure = copy.deepcopy(self.structures[model])  # Make a copy of the structure

            # Add points to structure
            structure = addPointsToStructure(structure, points, chain_id, residue_name)

            # Define output file path
            output_file = f"{output_folder}/{model}.pdb"

            # Save the modified structure to a PDB file
            _saveStructureToPDB(structure, output_file)

    def setUpSiteMapForModels(
        self,
        job_folder,
        target_residues,
        site_box=10,
        enclosure=0.5,
        maxvdw=1.1,
        resolution="fine",
        reportsize=100,
        overwrite=False,
        maxdist=8.0,
        sidechain=True,
        only_models=None,
        replace_symbol=None,
        write_conect_lines=True,
    ):
        """
        Generates a SiteMap calculation for model poses (no ligand) near specified residues.
        Parameters
        ==========
        job_folder : str
            Path to the calculation folder
        target_residues : dict
            Dictionary per model with a list of lists of residues (chain_id, residues) for which
            to calculate sitemap pockets.
        replace_symbol : str
            Symbol to replace for saving the models
        """

        # Create site map job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")

        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Copy script to generate protein and ligand mae inputs, separately.
        _copyScriptFile(job_folder, "prepareForSiteMap.py")
        script_path = job_folder + "/._prepareForSiteMap.py"

        # Save all input models
        self.saveModels(
            job_folder + "/input_models",
            write_conect_lines=write_conect_lines,
            replace_symbol=replace_symbol,
        )

        # Create input files
        jobs = []
        for model in self.models_names:

            # Skip models not in only_models list
            if only_models != None:
                if model not in only_models:
                    continue

            if replace_symbol:
                model_name = model.replace(replace_symbol[0], replace_symbol[1])
            else:
                model_name = model

            # Create an output folder for each model
            output_folder = job_folder + "/output_models/" + model_name
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            # Generate input protein files
            input_protein = job_folder + "/input_models/" + model_name + ".pdb"

            input_mae = (
                job_folder
                + "/output_models/"
                + model_name
                + "/"
                + model_name
                + "_protein.mae"
            )
            if not os.path.exists(input_mae) or overwrite:
                command = "run " + script_path + " "
                command += input_protein + " "
                command += output_folder + " "
                command += "--protein_only "
                os.system(command)

            if not isinstance(target_residues, dict):
                raise ValueError("Problem: target_residues must be a dictionary!")

            if model not in target_residues:
                raise ValueError(
                    f"Problem: model {model} not found in target_residues dictionary!"
                )

            elif isinstance(target_residues[model], (str, tuple)):
                target_residues[model] = [target_residues[model]]

            elif isinstance(target_residues[model][0], (str, tuple)):
                target_residues[model] = [target_residues[model]]

            for residue_selection in target_residues[model]:

                label = ""
                for r in residue_selection:
                    label += "".join([str(x) for x in r]) + "_"
                label = label[:-1]

                # Create folder
                if not os.path.exists(output_folder + "/" + label):
                    os.mkdir(output_folder + "/" + label)

                # Add site map command
                command = (
                    "cd "
                    + job_folder
                    + "/output_models/"
                    + model_name
                    + "/"
                    + label
                    + "\n"
                )
                command += '"${SCHRODINGER}/sitemap" '
                command += "-j " + model_name + " "
                command += "-prot ../" + model_name + "_protein.mae" + " "
                command += "-sitebox " + str(site_box) + " "
                command += "-resolution " + str(resolution) + " "
                command += "-keepvolpts yes "
                command += "-keeplogs yes "
                # command += '-maxdist '+str(maxdist)+' '
                # command += '-enclosure '+str(enclosure)+' '
                # command += '-maxvdw '+str(maxvdw)+' '
                command += "-reportsize " + str(reportsize) + " "

                command += '-siteasl "'
                for r in residue_selection:
                    if isinstance(r, tuple) and len(r) == 2:
                        command += (
                            "(chain.name "
                            + str(r[0])
                            + " and res.num {"
                            + str(r[1])
                            + "} "
                        )
                    elif isinstance(r, str) and len(r) == 1:
                        command += '"(chain.name ' + str(r[0]) + " "
                    else:
                        raise ValueError("Incorrect residue definition!")
                    if sidechain:
                        command += "and not (atom.pt ca,c,n,h,o)) or "
                    else:
                        command += ") or "
                command = command[:-4] + '" '
                command += "-HOST localhost:1 "
                command += "-TMPLAUNCHDIR "
                command += "-WAIT\n"
                command += "cd ../../../..\n"
                jobs.append(command)

        return jobs

    def setUpSiteMapForLigands(
        self, job_folder, poses_folder, site_box=10, resolution="fine", overwrite=False
    ):
        """
        Generates a SiteMap calculation for Docking poses outputed by the extractDockingPoses()
        function.
        Parameters
        ==========
        job_folder : str
            Path to the calculation folder
        poses_folder : str
            Path to docking poses folder.
        """

        # Create site map job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")

        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Copy script to generate protein and ligand mae inputs, separately.
        _copyScriptFile(job_folder, "prepareForSiteMap.py")
        script_path = job_folder + "/._prepareForSiteMap.py"

        # Create input files
        jobs = []
        for model in os.listdir(poses_folder):
            if not os.path.isdir(poses_folder + "/" + model):
                continue
            if not os.path.exists(job_folder + "/input_models/" + model):
                os.mkdir(job_folder + "/input_models/" + model)
            if not os.path.exists(job_folder + "/output_models/" + model):
                os.mkdir(job_folder + "/output_models/" + model)

            for pose in os.listdir(poses_folder + "/" + model):
                if pose.endswith(".pdb"):
                    pose_name = pose.replace(".pdb", "")

                    # Generate input protein and ligand files
                    input_ligand = (
                        job_folder
                        + "/input_models/"
                        + model
                        + "/"
                        + pose_name
                        + "_ligand.mae"
                    )
                    input_protein = (
                        job_folder
                        + "/input_models/"
                        + model
                        + "/"
                        + pose_name
                        + "_protein.mae"
                    )
                    if (
                        not os.path.exists(input_ligand)
                        or not os.path.exists(input_protein)
                        or overwrite
                    ):
                        command = "run " + script_path + " "
                        command += poses_folder + "/" + model + "/" + pose + " "
                        command += job_folder + "/input_models/" + model
                        os.system(command)

                    # Write Site Map input file
                    with open(
                        job_folder
                        + "/output_models/"
                        + model
                        + "/"
                        + pose_name
                        + ".in",
                        "w",
                    ) as smi:
                        smi.write(
                            "PROTEIN ../../input_models/"
                            + model
                            + "/"
                            + pose_name
                            + "_protein.mae\n"
                        )
                        smi.write(
                            "LIGMAE ../../input_models/"
                            + model
                            + "/"
                            + pose_name
                            + "_ligand.mae\n"
                        )
                        smi.write("SITEBOX " + str(site_box) + "\n")
                        smi.write("RESOLUTION " + resolution + "\n")
                        smi.write("REPORTSIZE 100\n")
                        smi.write("KEEPVOLPTS yes\n")
                        smi.write("KEEPLOGS yes\n")

                    # Add site map command
                    command = "cd " + job_folder + "/output_models/" + model + "\n"
                    command += '"${SCHRODINGER}/sitemap" '
                    command += pose_name + ".in" + " "
                    command += "-HOST localhost:1 "
                    command += "-TMPLAUNCHDIR "
                    command += "-WAIT\n"
                    command += "cd ../../..\n"
                    jobs.append(command)
        return jobs

    def analyseSiteMapCalculation(
        self,
        sitemap_folder,
        failed_value=0,
        verbose=True,
        output_models=None,
        replace_symbol=None,
    ):
        """
        Extract score values from a site map calculation.
         Parameters
         ==========
         sitemap_folder : str
             Path to the site map calculation folder. See
             setUpSiteMapForModels()
        failed_value : None, float or int
            The value to put in the columns of failed siteMap calculations.
        output_models : str
            Folder to combine models with sitemap points

        Returns
        =======
        sitemap_data : pandas.DataFrame
            Site map pocket information.
        """

        def parseVolumeInfo(log_file):
            """
            Parse eval log file for site scores.
            Parameters
            ==========
            eval_log : str
                Eval log file from sitemap output
            Returns
            =======
            pocket_data : dict
                Scores for the given pocket
            """
            with open(log_file) as lf:
                c = False
                for l in lf:
                    if l.startswith("SiteScore"):
                        c = True
                        labels = l.split()
                        continue
                    if c:
                        values = [float(x) for x in l.split()]
                        pocket_data = {x: y for x, y in zip(labels, values)}
                        c = False
            return pocket_data

        def checkIfCompleted(log_file):
            """
            Check log file for calculation completition.
            Parameters
            ==========
            log_file : str
                Path to the standard sitemap log file.
            Returns
            =======
            completed : bool
                Did the simulation end correctly?
            """
            with open(log_file) as lf:
                for l in lf:
                    if "SiteMap successfully completed" in l:
                        return True
            return False

        def checkIfFound(log_file):
            """
            Check log file for found sites.
            Parameters
            ==========
            log_file : str
                Path to the standard sitemap log file.
            Returns
            =======
            found : bool
                Did the simulation end correctly?
            """
            found = True
            with open(log_file) as lf:
                for l in lf:
                    if "No sites found" in l or "no sites were found" in l:
                        found = False
            return found

        if (
            replace_symbol
            and not isinstance(replace_symbol, tuple)
            and not len(replace_symbol) == 2
        ):
            raise ValueError(
                "replace_symbol must be a tuple: (old_symbol,  new_symbol)"
            )

        sitemap_data = {}
        sitemap_data["Model"] = []
        sitemap_data["Pocket"] = []

        input_folder = sitemap_folder + "/input_models"
        output_folder = sitemap_folder + "/output_models"

        if output_models:
            if not os.path.exists(output_models):
                os.mkdir(output_models)

        for model in os.listdir(output_folder):

            if replace_symbol:
                model_name = model.replace(replace_symbol[1], replace_symbol[0])
            else:
                model_name = model

            for r in os.listdir(output_folder + "/" + model):

                # Check if chain or residue was given
                if len(r) == 1:
                    pocket_type = "chain"
                else:
                    pocket_type = "residue"

                if os.path.isdir(output_folder + "/" + model + "/" + r):
                    log_file = (
                        output_folder + "/" + model + "/" + r + "/" + model + ".log"
                    )
                    if os.path.exists(log_file):
                        completed = checkIfCompleted(log_file)
                    else:
                        if verbose:
                            message = (
                                "Log file for model %s and "
                                + pocket_type
                                + " %s was not found!\n" % (model, r)
                            )
                            message += "It seems the calculation has not run yet..."
                            print(message)
                        continue

                    if not completed:
                        if verbose:
                            print(
                                "There was a problem with model %s and "
                                + pocket_type
                                + " %s" % (model, r)
                            )
                        continue
                    else:
                        found = checkIfFound(log_file)
                        if not found:
                            if verbose:
                                print(
                                    "No sites were found for model %s and "
                                    + pocket_type
                                    + " %s" % (model, r)
                                )
                            continue

                    pocket = r
                    pocket_data = parseVolumeInfo(log_file)

                    sitemap_data["Model"].append(model_name)
                    sitemap_data["Pocket"].append(pocket)

                    for l in pocket_data:
                        sitemap_data.setdefault(l, [])
                        sitemap_data[l].append(pocket_data[l])

                    if output_models:
                        print("Storing Volume Points models at %s" % output_models)
                        input_file = input_folder + "/" + model + ".pdb"
                        volpoint_file = (
                            output_folder
                            + "/"
                            + model
                            + "/"
                            + r
                            + "/"
                            + model
                            + "_site_1_volpts.pdb"
                        )
                        if os.path.exists(volpoint_file):

                            istruct = _readPDB(model + "_input", input_file)
                            imodel = [x for x in istruct.get_models()][0]
                            vstruct = _readPDB(model + "_volpts", volpoint_file)
                            vpt_chain = PDB.Chain.Chain("V")
                            for r in vstruct.get_residues():
                                vpt_chain.add(r)
                            imodel.add(vpt_chain)

                            _saveStructureToPDB(
                                istruct, output_models + "/" + model_name + "_vpts.pdb"
                            )
                        else:
                            print(
                                "Volume points PDB not found for model %s and residue %s"
                                % (m, r)
                            )

        sitemap_data = pd.DataFrame(sitemap_data)
        sitemap_data.set_index(["Model", "Pocket"], inplace=True)

        return sitemap_data

    def definePocketResiduesWithSiteMap(
        self,
        sitemap_folder,
        distance_to_points=2.5,
        only_models=None,
        output_file=None,
        overwrite=False,
        replace_symbol=None,
        sidechain_only=False,
    ):
        """
        Calculates the active site residues based on the volume points from a sitemap
        calcualtion. The models should be written with the option output_models from
        the analiseSiteMapCalculation() function.

        Parameters
        ==========
        sitemap_folder : str
            Path to the folder where sitemap calculation, containing the sitemap volume points residues, is located.
        only_models : (str, list)
            Specific models to be processed, if None all the models loaded in this class
            will be used
        distance_to_points : float
            The distance to consider a residue in contact with the volume point atoms.
        output_file : str
            Path the json output file to store the residue data
        overwrite : bool
            Overwrite json file if found? (essentially, calculate all again)
        """

        def merge_pdbs_one_model(pdb1, pdb2, output_pdb):
            parser = PDB.PDBParser(QUIET=True)
            io = PDB.PDBIO()

            # Load both PDB structures
            struct1 = parser.get_structure("struct1", pdb1)
            struct2 = parser.get_structure("struct2", pdb2)

            # Create a new structure and a new model (single frame)
            merged_struct = PDB.Structure.Structure("merged")
            merged_model = PDB.Model.Model(0)  # Single model with ID 0

            # Add all chains from the first structure's first model
            for chain in struct1[0]:
                merged_model.add(chain)

            # Add all chains from the second structure's first model
            for chain in struct2[0]:
                # Optionally, rename chain IDs if there's a conflict:
                # chain.id = chain.id + "_2"
                merged_model.add(chain)

            # Add the merged model to the new structure
            merged_struct.add(merged_model)

            # Write the merged structure to file
            io.set_structure(merged_struct)
            io.save(output_pdb)

        if output_file == None:
            raise ValueError("An ouput file name must be given")
        if not output_file.endswith(".json"):
            output_file = output_file + ".json"

        if (
            replace_symbol
            and not isinstance(replace_symbol, tuple)
            and not len(replace_symbol) == 2
        ):
            raise ValueError(
                "replace_symbol must be a tuple: (old_symbol,  new_symbol)"
            )

        if not os.path.exists(output_file) or overwrite:

            residues = {}
            for model in self:

                # Skip models not in only_models list
                if only_models != None:
                    if model not in only_models:
                        continue

                if replace_symbol:
                    model_name = model.replace(replace_symbol[0], replace_symbol[1])
                else:
                    model_name = model

                # Get input PDB
                input_pdb = f'{sitemap_folder}/input_models/{model}.pdb'

                if not os.path.exists(input_pdb):
                    raise ValueError(f'Input file {input_pdb} not found!')

                residues.setdefault(model, {})

                # Check if the volume points model file exists
                output_path = f'{sitemap_folder}/output_models/{model}'

                for resp in os.listdir(output_path):

                    if not bool(re.search(r'\d+$', resp)):
                        continue

                    resp_path = output_path+"/"+resp
                    volpts_file = None
                    for pdb in os.listdir(resp_path):
                        if pdb.endswith('volpts.pdb'):
                            volpts_file =  resp_path+"/"+pdb

                    if not volpts_file:
                        print(
                            "Model %s not found in the volume points folder %s!"
                            % (model, resp_path)
                        )
                        continue

                    # Combine input and volpts pdbs
                    tmp_file = volpts_file.replace('.pdb', '.tmp.pdb')
                    merge_pdbs_one_model(input_pdb, volpts_file, tmp_file)

                    # Compute neighbours
                    traj = md.load(tmp_file)
                    os.remove(tmp_file)

                    if sidechain_only:
                        protein = traj.topology.select("protein and sidechain and not resname vpt")
                    else:
                        protein = traj.topology.select("protein and not resname vpt")

                    vpts = traj.topology.select("resname vpt")
                    n = md.compute_neighbors(
                        traj, distance_to_points / 10, vpts, haystack_indices=protein
                    )
                    residues[model][resp] = list(
                        set(
                            [
                                traj.topology.atom(i).residue.resSeq
                                for i in n[0]
                                if traj.topology.atom(i).is_sidechain
                            ]
                        )
                    )

            with open(output_file, "w") as jf:
                json.dump(residues, jf)

        else:
            with open(output_file) as jf:
                residues = json.load(jf)

        for model in residues:
            for residue in residues[model]:
                residues[model][residue] = np.array(list(residues[model][residue]))

        return residues

    def getInContactResidues(
        self,
        residue_selection,
        distance_threshold=2.5,
        sidechain_selection=False,
        return_residues=False,
        only_protein=False,
        sidechain=False,
        backbone=False,
    ):
        """
        Get residues in close contact to a residue selection
        """

        in_contact = {}
        for model in self:

            # Get structure coordinates
            structure = self.structures[model]
            selected_coordinates = _getStructureCoordinates(
                structure,
                sidechain=sidechain_selection,
                only_residues=residue_selection[model],
            )
            selected_atoms = _getStructureCoordinates(
                structure,
                sidechain=sidechain_selection,
                only_residues=residue_selection[model],
                return_atoms=True,
            )

            other_coordinates = _getStructureCoordinates(
                structure,
                sidechain=sidechain,
                exclude_residues=residue_selection[model],
            )
            other_atoms = _getStructureCoordinates(
                structure,
                sidechain=sidechain,
                exclude_residues=residue_selection[model],
                return_atoms=True,
            )

            if selected_coordinates.size == 0:
                raise ValueError(
                    f"Problem matching the given residue selection for model {model}"
                )

            # Compute the distance matrix between the two set of coordinates
            M = distance_matrix(selected_coordinates, other_coordinates)
            in_contact[model] = np.array(other_atoms)[
                np.argwhere(M <= distance_threshold)[:, 1]
            ]
            in_contact[model] = [tuple(a) for a in in_contact[model]]

            # Only return tuple residues
            if return_residues:
                residues = []
                for atom in in_contact[model]:
                    residues.append(tuple(atom[:2]))
                in_contact[model] = list(set(residues))

        return in_contact

    def setUpLigandParameterization(
        self,
        job_folder,
        ligands_folder,
        charge_method=None,
        only_ligands=None,
        rotamer_resolution=10,
    ):
        """
        Run PELE platform for ligand parameterization
        Parameters
        ==========
        job_folder : str
            Path to the job input folder
        ligands_folder : str
            Path to the folder containing the ligand molecules in PDB format.
        """

        charge_methods = ["gasteiger", "am1bcc", "OPLS"]
        if charge_method == None:
            charge_method = "OPLS"

        if charge_method not in charge_methods:
            raise ValueError(
                "The charge method should be one of: " + str(charge_methods)
            )

        # Create PELE job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        # Copy script to generate protein and ligand mae inputs, separately.
        _copyScriptFile(job_folder, "peleffy_ligand.py")

        jobs = []
        for ligand in os.listdir(ligands_folder):

            extension = ligand.split(".")[-1]

            if extension == "pdb":
                ligand_name = ligand.replace("." + extension, "")

                # Only process ligands given in only_ligands list
                if only_ligands != None:
                    if ligand_name not in only_ligands:
                        continue

                # structure = _readPDB(ligand_name, ligands_folder+'/'+ligand)
                if not os.path.exists(job_folder + "/" + ligand_name):
                    os.mkdir(job_folder + "/" + ligand_name)

                # _saveStructureToPDB(structure, job_folder+'/'+pdb_name+'/'+pdb_name+extension)
                shutil.copyfile(
                    ligands_folder + "/" + ligand,
                    job_folder
                    + "/"
                    + ligand_name
                    + "/"
                    + ligand_name
                    + "."
                    + extension,
                )

                # Create command
                command = "cd " + job_folder + "/" + ligand_name + "\n"
                command += (
                    "python  ../._peleffy_ligand.py "
                    + ligand_name
                    + "."
                    + extension
                    + " "
                )
                command += "--rotamer_resolution " + str(rotamer_resolution) + " "
                command += "\n"
                command += "cd ../..\n"
                jobs.append(command)

        return jobs

    def _setUpCovalentLigandParameterization(
        self, model, residue_index, base_aa, output_folder=""
    ):
        """
        Add a step of parameterization for a covalent residue in a specific model.

        Parameters
        ==========
        model : str
            Model name
        resname : str
            Name of the covalent residue
        base_aa : dict
            Three letter identity of the aminoacid upon which the ligand is covalently bound.
            One entry in the dictionary for residue name, i.e., base_aa={'FAD':'CYS', 'NAD':'ASP', etc.}
        output_folder : str
            Output folder where to put the ligand PDB file.
        """

        def getAtomsIndexes(pdb_file):

            atoms_indexes = {}
            with open(pdb_file) as pdb:
                for l in pdb:
                    if l.startswith("ATOM") or l.startswith("HETATM"):
                        index, name, chain, resid = (
                            int(l[7:12]),
                            l[12:17].strip(),
                            l[21],
                            int(l[22:27]),
                        )
                        atoms_indexes[(chain, resid, name)] = index
            return atoms_indexes

        def addConectLines(model, pdb_file):

            # Add conect lines to ligand structure
            atoms_indexes = getAtomsIndexes(pdb_file)
            with open(pdb_file) as pdb:
                for l in pdb:
                    if l.startswith("ATOM") or l.startswith("HETATM"):
                        index, name, chain, resid = (
                            int(l[7:12]),
                            l[12:17].strip(),
                            l[21],
                            int(l[22:27]),
                        )
                        atoms_indexes[(chain, resid, name)] = index

            with open(pdb_file + ".tmp", "w") as tmp:
                with open(pdb_file) as pdb:
                    for l in pdb:
                        if l.startswith("ATOM") or l.startswith("HETATM"):
                            if not l.startswith("END"):
                                tmp.write(l)

                    # count = 0
                    for conect in self.conects[model]:
                        if conect[0] in atoms_indexes:
                            line = "CONECT"
                            for atom in conect:
                                if atom in atoms_indexes:
                                    line += "%5s" % atoms_indexes[atom]
                            line += "\n"
                            tmp.write(line)
                            # count += 1

                tmp.write("END\n")
            shutil.move(pdb_file + ".tmp", pdb_file)

        # Define atom names
        c_atom = "C"
        n_atom = "N"
        o_atom = "O"

        ### Create covalent-ligand-only PDB
        cov_structure = PDB.Structure.Structure(0)
        cov_model = PDB.Model.Model(0)
        cov_chain = None
        for r in self.structures[model].get_residues():
            if r.id[1] == residue_index:
                resname = r.resname
                if resname not in base_aa:
                    message = "Residue %s not found in the base_aa dictionary!"
                    message += "Please give the base of the aminoacid with the 'base_aa' keyword"
                    raise ValueError(message)

                cov_residue = r
                cov_chain = PDB.Chain.Chain(r.get_parent().id)
                cov_chain.add(r)
                break

        if cov_chain == None:
            raise ValueError(
                "Residue %s not found in model %s structure" % (resname, model)
            )

        cov_model.add(cov_chain)
        cov_structure.add(cov_model)
        _saveStructureToPDB(cov_structure, output_folder + "/" + resname + ".pdb")
        addConectLines(model, output_folder + "/" + resname + ".pdb")

        # Get atoms to which append hydrogens
        indexes = getAtomsIndexes(output_folder + "/" + resname + ".pdb")
        selected_atoms = []
        for atom in indexes:
            if atom[-1] == c_atom:
                c_atom = str(indexes[atom])
                selected_atoms.append(c_atom)
            elif atom[-1] == n_atom:
                n_atom = str(indexes[atom])
                selected_atoms.append(n_atom)
            elif atom[-1] == o_atom:
                o_atom = str(indexes[atom])
        selected_atoms = ",".join(selected_atoms)

        # Set C-O bond as double bond to secure single hydrogen addition to C atom
        add_bond = str(c_atom) + "," + str(o_atom) + ",2"

        _copyScriptFile(output_folder, "addHydrogens.py")

        ### Add hydrogens to PDB structure
        print("Replacing covalent bonds with hydrogens at %s residue..." % resname)
        command = "run python3 " + output_folder + "/._addHydrogens.py "
        command += output_folder + "/" + resname + ".pdb "
        command += output_folder + "/" + resname + ".pdb "
        command += "--indexes " + selected_atoms + " "
        command += "--add_bond " + add_bond + " "
        command += "--covalent"
        os.system(command)

        # Copy file to avoid the faulty preparation...
        shutil.copyfile(
            output_folder + "/" + resname + ".pdb",
            output_folder + "/" + resname + "_p.pdb",
        )

    def setUpPELECalculation(
        self,
        pele_folder,
        models_folder,
        input_yaml,
        box_centers=None,
        distances=None,
        angles=None,
        constraints=None,
        ligand_index=1,
        box_radius=10,
        steps=100,
        debug=False,
        iterations=5,
        cpus=96,
        equilibration_steps=100,
        ligand_energy_groups=None,
        separator="-",
        use_peleffy=True,
        usesrun=True,
        energy_by_residue=False,
        ebr_new_flag=False,
        ninety_degrees_version=False,
        analysis=False,
        energy_by_residue_type="all",
        peptide=False,
        equilibration_mode="equilibrationLastSnapshot",
        spawning="independent",
        continuation=False,
        equilibration=True,
        skip_models=None,
        skip_ligands=None,
        extend_iterations=False,
        only_models=None,
        only_ligands=None,
        only_combinations=None,
        ligand_templates=None,
        seed=12345,
        log_file=False,
        nonbonded_energy=None,
        nonbonded_energy_type="all",
        nonbonded_new_flag=False,
        covalent_setup=False,
        covalent_base_aa=None,
        membrane_residues=None,
        bias_to_point=None,
        com_bias1=None,
        com_bias2=None,
        com_residue_pairs={},
        epsilon=0.5,
        rescoring=False,
        ligand_equilibration_cst=True,
        regional_metrics=None,
        regional_thresholds=None,
        regional_combinations=None,
        regional_exclusions=None,
        max_regional_iterations=None,
        regional_energy_bias="Binding Energy",
        regional_best_fraction=0.2,
        constraint_level=1,
        restore_input_coordinates=False,
        skip_connect_rewritting=False
    ):
        """
        Generates a PELE calculation for extracted poses. The function reads all the
        protein-ligand poses and creates input files for a PELE simulation setup.

        Parameters
        ----------
        pele_folder : str
            Path to the folder where PELE calculations will be located.
        models_folder : str
            Path to the folder containing input docking poses.
        input_yaml : str
            Path to the input YAML file used as a template for all runs.
        box_centers : dict, optional
            Dictionary specifying the centers of the simulation box for each model.
        distances : dict, optional
            Distance constraints for the simulation. Format:
            {model_name: [(chain1, residue1, atom1), (chain2, residue2, atom2)]}.
        angles : dict, optional
            Angular constraints for the simulation.
        constraints : dict, optional
            Positional and distance constraints for each model and ligand.
        ligand_index : int, optional, default=1
            Index of the ligand in the structure file.
        box_radius : float, optional, default=10
            Radius of the simulation box.
        steps : int, optional, default=100
            Number of simulation steps per iteration.
        debug : bool, optional, default=False
            If True, enables debug mode for the simulation.
        iterations : int, optional, default=5
            Number of iterations for the simulation.
        cpus : int, optional, default=96
            Number of CPUs to use for parallelization.
        equilibration_steps : int, optional, default=100
            Number of equilibration steps before the production run.
        ligand_energy_groups : dict, optional
            Additional groups to consider for energy by residue reports.
        separator : str, optional, default="-"
            Separator used in filenames between protein and ligand identifiers.
        use_peleffy : bool, optional, default=True
            If True, PELEffy will be used for ligand parameterization.
        usesrun : bool, optional, default=True
            If True, the PELE simulation will use srun for job submission.
        energy_by_residue : bool, optional, default=False
            If True, energy by residue will be calculated.
        ebr_new_flag : bool, optional, default=False
            If True, uses the new version of the energy by residue calculation.
        ninety_degrees_version : bool, optional, default=False
            If True, uses the 90 degrees version of PELE.
        analysis : bool, optional, default=False
            If True, enables analysis mode after the simulation.
        energy_by_residue_type : str, optional, default="all"
            Type of energy to be calculated per residue. Options: 'all', 'lennard_jones', 'sgb', 'electrostatic'.
        peptide : bool, optional, default=False
            If True, treats the system as a peptide for specific setup steps.
        equilibration_mode : str, optional, default="equilibrationLastSnapshot"
            Mode used for equilibration: "equilibrationLastSnapshot" or "equilibrationCluster".
        spawning : str, optional, default="independent"
            Spawning method used for adaptive sampling. Options include 'independent', 'epsilon', 'variableEpsilon', etc.
        continuation : bool, optional, default=False
            If True, continues from the previous run instead of starting fresh.
        equilibration : bool, optional, default=True
            If True, performs equilibration before the production run.
        skip_models : list, optional
            List of protein models to skip from the simulation.
        skip_ligands : list, optional
            List of ligands to skip from the simulation.
        extend_iterations : bool, optional, default=False
            If True, extends the number of iterations beyond the default.
        only_models : list, optional
            List of protein models to include in the simulation.
        only_ligands : list, optional
            List of ligands to include in the simulation.
        only_combinations : list, optional
            List of protein-ligand combinations to include.
        ligand_templates : str, optional
            Path to custom ligand templates for parameterization.
        seed : int, optional, default=12345
            Random seed for the simulation.
        log_file : bool, optional, default=False
            If True, enables logging to a file.
        nonbonded_energy : dict, optional
            Dictionary specifying nonbonded energy atoms for specific protein-ligand pairs.
        nonbonded_energy_type : str, optional, default="all"
            Type of nonbonded energy to calculate. Options: 'all', 'lennard_jones', 'electrostatic'.
        nonbonded_new_flag : bool, optional, default=False
            If True, uses the new version of nonbonded energy calculation.
        covalent_setup : bool, optional, default=False
            If True, sets up the simulation for covalently bound ligands.
        covalent_base_aa : dict, optional
            Dictionary specifying the amino acid residue involved in covalent binding for each ligand.
        membrane_residues : dict, optional
            Dictionary specifying membrane residues to apply specific constraints.
        bias_to_point : dict, optional
            Dictionary specifying biasing points in the system for specific models.
        com_bias1 : dict, optional
            First group of atoms for center-of-mass biasing.
        com_bias2 : dict, optional
            Second group of atoms for center-of-mass biasing.
        epsilon : float, optional, default=0.5
            Epsilon value used for biasing the center-of-mass distance.
        rescoring : bool, optional, default=False
            If True, performs a rescoring calculation.
        ligand_equilibration_cst : bool, optional, default=True
            If True, applies constraints during ligand equilibration.
        regional_metrics : dict, optional
            Metrics for regional spawning in adaptive sampling.
        regional_thresholds : dict, optional
            Thresholds for regional spawning metrics.
        max_regional_iterations : int, optional
            Maximum number of iterations for regional spawning.
        regional_energy_bias : str, optional, default="Binding Energy"
            Bias metric for regional spawning, either 'Total Energy' or 'Binding Energy'.
        regional_best_fraction : float, optional, default=0.2
            Fraction of the best-performing states selected in regional spawning.
        constraint_level : int, optional, default=1
            Level of constraints applied during the simulation (0 for none, 1 for basic).
        restore_input_coordinates : bool, optional, default=False
            If True, restores the original coordinates after PELE processing (not working)

        Returns
        -------
        list
            A list of shell commands to run the PELE jobs.

        Detailed input examples
        -----------------------
        com_group_1 : list of lists
            A list containing atom definitions for the first group of atoms used in center-of-mass (COM) distance calculations.
            Each atom is defined as a list with the following format:

            [
                [chain_id, residue_id, atom_name],
                [chain_id, residue_id, atom_name],
                ...
            ]

            Where:
            - chain_id : str
                The ID of the chain where the atom is located (e.g., 'A', 'B').
            - residue_id : int
                The residue number where the atom is located (e.g., 100).
            - atom_name : str
                The name of the atom (e.g., 'CA' for alpha carbon, 'O' for oxygen).

            Example:
            [
                ["A", 100, "CA"],
                ["A", 102, "CB"],
                ["B", 150, "O"]
            ]

            In this example, the group includes:
            - The alpha carbon atom (CA) of residue 100 in chain 'A'.
            - The beta carbon atom (CB) of residue 102 in chain 'A'.
            - The oxygen atom (O) of residue 150 in chain 'B'.

        com_group_2 : list of lists
            The format is identical to com_group_1 but represents the second group of atoms.
            These two groups are used to calculate the center-of-mass distance between them.
        """

        # Flag for checking if continuation was given as True
        if continuation:
            continue_all = True
        else:
            continue_all = False

        energy_by_residue_types = ["all", "lennard_jones", "sgb", "electrostatic"]
        if energy_by_residue_type not in energy_by_residue_types:
            raise ValueError(
                "%s not found. Try: %s"
                % (energy_by_residue_type, energy_by_residue_types)
            )

        spawnings = [
            "independent",
            "inverselyProportional",
            "epsilon",
            "variableEpsilon",
            "independentMetric",
            "UCB",
            "FAST",
            "ProbabilityMSM",
            "MetastabilityMSM",
            "IndependentMSM",
            "regional",
        ]

        methods = ["rescoring"]

        if spawning != None and spawning not in spawnings:
            message = "Spawning method %s not found." % spawning
            message = "Allowed options are: " + str(spawnings)
            raise ValueError(message)

        regional_spawning = False
        if spawning == "regional":

            # Check for required inputs
            if not isinstance(regional_metrics, dict):
                raise ValueError(
                    "For the regional spawning you must define the regional_metrics dictionary."
                )
            if not isinstance(regional_thresholds, dict):
                raise ValueError(
                    "For the regional spawning you must define the regional_thresholds dictionary."
                )

            if regional_energy_bias not in ["Total Energy", "Binding Energy"]:
                raise ValueError(
                    'You must give either "Total Energy" or "Binding Energy" to bias the regional spawning simulation!'
                )

            if (regional_combinations or regional_exclusions) and not (regional_combinations and regional_exclusions):
                raise ValueError('You must give both, regional_combinations and regional_exclusions not just one of them.')

            regional_spawning = True
            spawning = "independent"

        if isinstance(membrane_residues, type(None)):
            membrane_residues = {}

        if isinstance(bias_to_point, type(None)):
            bias_to_point = {}

        # Check bias_to_point input
        if isinstance(bias_to_point, (list, tuple)):
            d = {}
            for model in self:
                d[model] = bias_to_point
            bias_to_point = d

        if not isinstance(bias_to_point, dict):
            raise ValueError("bias_to_point should be a dictionary or a list.")

        # Check COM distance bias inputs
        if isinstance(com_bias1, type(None)):
            com_bias1 = {}

        if isinstance(com_bias2, type(None)):
            com_bias2 = {}

        if com_bias1 != {} and com_bias2 == {} or com_bias1 == {} and com_bias2 != {}:
            raise ValueError(
                "You must give both COM atom groups to apply a COM distance bias."
            )

        if isinstance(com_residue_pairs, type(None)):
            com_residue_pairs = {}

        # Create PELE job folder
        if not os.path.exists(pele_folder):
            os.mkdir(pele_folder)

        # Use to find the relative location of general scripts
        rel_path_to_root = "../"
        if regional_spawning:
            rel_path_to_root = "../" * 2
            _copyScriptFile(pele_folder, "regionalSpawning.py")

        # Read docking poses information from models_folder and create pele input
        # folders.
        jobs = []
        models = {}
        ligand_pdb_name = {}
        pose_number = {}
        for d in os.listdir(models_folder):
            if os.path.isdir(models_folder + "/" + d):
                for f in os.listdir(models_folder + "/" + d):

                    fs = f.split(separator)
                    protein = fs[0]
                    ligand = fs[1]
                    pose = fs[2].replace(".pdb", "")
                    pose_number[(protein, ligand)] = pose

                    # Skip given protein models
                    if skip_models != None:
                        if protein in skip_models:
                            continue

                    # Skip given ligand models
                    if skip_ligands != None:
                        if ligand in skip_ligands:
                            continue

                    # Skip proteins not in only_proteins list
                    if only_models != None:
                        if protein not in only_models:
                            continue

                    # Skip proteins not in only_ligands list
                    if only_ligands != None:
                        if ligand not in only_ligands:
                            continue

                    if only_combinations and (protein, ligand) not in only_combinations:
                        continue

                    # Create PELE job folder for each docking
                    protein_ligand = protein + separator + ligand
                    protein_ligand_folder = pele_folder + "/" + protein_ligand
                    if not os.path.exists(protein_ligand_folder):
                        os.mkdir(protein_ligand_folder)

                    if regional_spawning:

                        # Create metrics dictionaries
                        reg_met = {}
                        metric_types = {}
                        for m in regional_metrics:
                            if protein not in regional_metrics[m]:
                                raise ValueError(
                                    f"Protein {protein} was not found in the regional_metrics dictionary for metric {m}"
                                )

                            if ligand not in regional_metrics[m][protein]:
                                raise ValueError(
                                    f"Ligand {ligand} was not found in the regional_metrics dictionary for protein {protein} and metric {m}"
                                )

                            # Check if distance_ and angle_ prefix were given
                            reg_met[m] = []
                            for v in regional_metrics[m][protein][ligand]:
                                if "-" in v:
                                    v = v.replace("-", "_")
                                if not v.startswith("distance_") and not v.startswith(
                                    "angle_"
                                ):
                                    if len(v.split("_")) == 2:
                                        prefix = "distance"
                                    elif len(v.split("_")) == 3:
                                        prefix = "angle"
                                    v =  prefix+'_'+v
                                else:
                                    prefix = v.split('_')[0]
                                reg_met[m].append(v)
                                metric_types.setdefault(m, prefix)

                        with open(protein_ligand_folder + "/metrics.json", "w") as jf:
                            json.dump(reg_met, jf)

                        # Check regional thresholds format
                        for m in regional_thresholds:
                            rm = regional_thresholds[m]

                            incorrect = False
                            if not isinstance(rm, (int, float)) and not isinstance(
                                rm, tuple
                            ):
                                incorrect = True
                            elif isinstance(rm, tuple) and len(rm) != 2:
                                incorrect = True
                            elif isinstance(rm, tuple) and (
                                not isinstance(rm[0], (int, float))
                                or not isinstance(rm[1], (int, float))
                            ):
                                incorrect = True
                            if incorrect:
                                raise ValueError(
                                    "The regional thresholds should be floats or two-elements tuples of floats"
                                )  # Review this check for more complex region definitions

                        with open(
                            protein_ligand_folder + "/metrics_thresholds.json", "w"
                        ) as jf:
                            json.dump(regional_thresholds, jf)

                        if regional_combinations:

                            # Collect all unique metrics from combinations
                            unique_metrics = set()
                            for new_metric, metrics in regional_combinations.items():
                                metric_metric_types = [metric_types[m] for m in reg_met]
                                if len(set(metric_metric_types)) != 1:
                                    raise ValueError('For regional spawning, you are attempting to combine different metric types (e.g., distances and angles) is not allowed.')
                                unique_metrics.update(metrics)
                                metrics_list = list(unique_metrics)

                            # Ensure all required metric columns were given in the regional metrics list
                            missing_columns = set(metrics_list) - set(regional_metrics.keys())
                            if missing_columns:
                                raise ValueError(f"Missing combination metrics in regional metrics: {missing_columns}")

                            # Check all exclusion metrics are defined in the combinations metrics
                            excusion_metrics = []

                            for exclusion in regional_exclusions:
                                if isinstance(regional_exclusions, list):
                                    excusion_metrics += [x for x in exclusion]
                                elif isinstance(regional_exclusions, dict):
                                    excusion_metrics += [x for x in regional_exclusions[exclusion]]

                            missing_columns = set(excusion_metrics) - set(metrics_list)
                            if missing_columns:
                                raise ValueError(f"Missing exclusion metrics in combination metrics: {missing_columns}")

                            with open(
                                protein_ligand_folder + "/regional_combinations.json", "w"
                            ) as jf:
                                json.dump(regional_combinations, jf)

                            with open(
                                protein_ligand_folder + "/regional_exclusions.json", "w"
                            ) as jf:
                                json.dump(regional_exclusions, jf)

                        protein_ligand_folder = protein_ligand_folder + "/0"
                        if not os.path.exists(protein_ligand_folder):
                            os.mkdir(protein_ligand_folder)

                    structure = _readPDB(
                        protein_ligand, models_folder + "/" + d + "/" + f
                    )

                    # Change water names if any
                    for residue in structure.get_residues():
                        if residue.id[0] == "W":
                            residue.resname = "HOH"

                        if residue.get_parent().id == "L":
                            ligand_pdb_name[ligand] = residue.resname

                    ## Add dummy atom if peptide docking ### Strange fix =)
                    if peptide:
                        for chain in structure.get_chains():
                            if chain.id == "L":
                                # Create new residue
                                new_resid = (
                                    max([r.id[1] for r in chain.get_residues()]) + 1
                                )
                                residue = PDB.Residue.Residue(
                                    ("H", new_resid, " "), "XXX", " "
                                )
                                serial_number = (
                                    max([a.serial_number for a in chain.get_atoms()])
                                    + 1
                                )
                                atom = PDB.Atom.Atom(
                                    "X",
                                    [0, 0, 0],
                                    0,
                                    1.0,
                                    " ",
                                    "%-4s" % "X",
                                    serial_number + 1,
                                    "H",
                                )
                                residue.add(atom)
                                chain.add(residue)

                    if skip_connect_rewritting:
                        print(f'The structure {f} has pre-defined CONECT lines probably from extractPELEPoses() function. Skipping saving structure and re-writting them. Directly copying structure to PELE folder..')
                        # Specify the source file path
                        source_file = f'{models_folder}/{d}/{f}'
                        # Specify the destination file path
                        destination_folder = f'{protein_ligand_folder}/{f}'
                        # Perform the copy operation
                        print(f'Copying the structure {f} from source folder: {models_folder}/{d}/{f} to destination_folder: {protein_ligand_folder}')
                        shutil.copyfile(source_file, destination_folder)

                    else:
                        _saveStructureToPDB(structure, protein_ligand_folder + "/" + f)
                        self._write_conect_lines(
                            protein, protein_ligand_folder + "/" + f, check_file=True
                        )

                    if (protein, ligand) not in models:
                        models[(protein, ligand)] = []
                    models[(protein, ligand)].append(f)

                    # If templates are given for ligands
                    templates = {}
                    if ligand_templates != None:

                        # Create templates folder
                        if not os.path.exists(pele_folder + "/templates"):
                            os.mkdir(pele_folder + "/templates")

                        for ligand in os.listdir(ligand_templates):

                            if not os.path.isdir(ligand_templates + "/" + ligand):
                                continue

                            # Create ligand template folder
                            if not os.path.exists(pele_folder + "/templates/" + ligand):
                                os.mkdir(pele_folder + "/templates/" + ligand)

                            templates[ligand] = []
                            for f in os.listdir(ligand_templates + "/" + ligand):
                                if f.endswith(".rot.assign") or f.endswith("z"):

                                    # Copy template files
                                    shutil.copyfile(
                                        ligand_templates + "/" + ligand + "/" + f,
                                        pele_folder + "/templates/" + ligand + "/" + f,
                                    )

                                    templates[ligand].append(f)

        # Create YAML file
        for model in models:
            protein, ligand = model
            protein_ligand = protein + separator + ligand
            protein_ligand_folder = pele_folder + "/" + protein_ligand
            pose = pose_number[model]
            if regional_spawning:
                protein_ligand_folder += "/0"

            keywords = [
                "system",
                "chain",
                "resname",
                "steps",
                "iterations",
                "atom_dist",
                "analyse",
                "cpus",
                "equilibration",
                "equilibration_steps",
                "traj",
                "working_folder",
                "usesrun",
                "use_peleffy",
                "debug",
                "box_radius",
                "box_center",
                "equilibration_mode",
                "seed",
                "spawning",
                "constraint_level",
            ]

            # Generate covalent parameterization setup
            if not covalent_setup:
                if protein in self.covalent and self.covalent[protein] != []:
                    print(
                        "WARNING: Covalent bound ligands were found. Consider giving covalent_setup=True"
                    )
            else:
                if covalent_base_aa == None:
                    message = (
                        "You must give the base AA upon which each covalently"
                    )
                    message += "attached ligand is bound. E.g., covalent_base_aa=base_aa={'FAD':'CYS', 'NAD':'ASP', etc.}"
                    raise ValueError(message)

                if protein in self.covalent:
                    for index in self.covalent[protein]:
                        output_folder = protein_ligand_folder + "/output/"
                        if not os.path.exists(output_folder + "/ligand"):
                            os.makedirs(output_folder + "/ligand")
                        self._setUpCovalentLigandParameterization(
                            protein,
                            index,
                            covalent_base_aa,
                            output_folder=output_folder + "/ligand",
                        )

                        # Copy covalent parameterization script
                        _copyScriptFile(
                            output_folder, "covalentLigandParameterization.py"
                        )

                        # Define covalent parameterization command
                        skip_covalent_residue = [
                            r.resname
                            for r in self.structures[protein].get_residues()
                            if r.id[1] == index
                        ][0]
                        covalent_command = "cd output\n"
                        covalent_command += (
                            "python "
                            + rel_path_to_root
                            + "._covalentLigandParameterization.py ligand/"
                            + skip_covalent_residue
                            + ".pdb "
                            + skip_covalent_residue
                            + " "
                            + covalent_base_aa[skip_covalent_residue]
                            + "\n"
                        )

                        # Copy modify processed script
                        _copyScriptFile(
                            output_folder, "modifyProcessedForCovalentPELE.py"
                        )
                        cov_residues = ",".join(
                            [str(x) for x in self.covalent[protein]]
                        )
                        covalent_command += (
                            "python "
                            + rel_path_to_root
                            + "._modifyProcessedForCovalentPELE.py "
                            + cov_residues
                            + " \n"
                        )
                        covalent_command += "mv DataLocal/Templates/OPLS2005/Protein/templates_generated/* DataLocal/Templates/OPLS2005/Protein/\n"
                        covalent_command += "cd ..\n"

            # Write input yaml
            with open(protein_ligand_folder + "/" + "input.yaml", "w") as iyf:
                if energy_by_residue or nonbonded_energy != None:
                    # Use new PELE version with implemented energy_by_residue
                    iyf.write('pele_exec: "/gpfs/projects/bsc72/PELE++/mnv/1.8.1b1/bin/PELE_mpi"\n')
                    iyf.write('pele_data: "/gpfs/projects/bsc72/PELE++/mnv/1.8.1b1/Data"\n')
                    iyf.write('pele_documents: "/gpfs/projects/bsc72/PELE++/mnv/1.8.1b1/Documents/"\n')
                elif ninety_degrees_version:
                    # Use new PELE version with implemented 90 degrees fix
                    print('paths of PELE version should be changed')
                    iyf.write(
                        'pele_exec: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/bin/PELE-1.8_mpi"\n'
                    )
                    iyf.write(
                        'pele_data: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Data"\n'
                    )
                    iyf.write(
                        'pele_documents: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Documents/"\n'
                    )
                if len(models[model]) > 1:
                    equilibration_mode = "equilibrationCluster"
                    iyf.write("system: '*.pdb'\n")
                else:
                    iyf.write("system: '" + " ".join(models[model]) + "'\n")
                iyf.write("chain: 'L'\n")
                if peptide:
                    iyf.write("resname: 'XXX'\n")
                    iyf.write("skip_ligand_prep:\n")
                    iyf.write(" - 'XXX'\n")
                else:
                    iyf.write("resname: '" + ligand_pdb_name[ligand] + "'\n")
                iyf.write("steps: " + str(steps) + "\n")
                iyf.write("iterations: " + str(iterations) + "\n")
                iyf.write("cpus: " + str(cpus) + "\n")
                if equilibration:
                    iyf.write("equilibration: true\n")
                    iyf.write(
                        "equilibration_mode: '" + equilibration_mode + "'\n"
                    )
                    iyf.write(
                        "equilibration_steps: "
                        + str(equilibration_steps)
                        + "\n"
                    )
                else:
                    iyf.write("equilibration: false\n")
                if spawning != None:
                    iyf.write("spawning: '" + str(spawning) + "'\n")
                if constraint_level:
                    iyf.write(
                        "constraint_level: " + str(constraint_level) + "\n"
                    )
                if rescoring:
                    iyf.write("rescoring: true\n")

                iyf.write("traj: trajectory.xtc\n")
                iyf.write("working_folder: 'output'\n")
                if usesrun:
                    iyf.write("usesrun: true\n")
                else:
                    iyf.write("usesrun: false\n")
                if use_peleffy:
                    iyf.write("use_peleffy: true\n")
                else:
                    iyf.write("use_peleffy: false\n")
                if analysis:
                    iyf.write("analyse: true\n")
                else:
                    iyf.write("analyse: false\n")

                if covalent_setup:
                    iyf.write("skip_ligand_prep:\n")
                    iyf.write(' - "' + skip_covalent_residue + '"\n')

                if ligand in templates:
                    iyf.write("templates:\n")
                    iyf.write(' - "LIGAND_TEMPLATE_PATH_ROT"\n')
                    iyf.write(' - "LIGAND_TEMPLATE_PATH_Z"\n')
                    iyf.write("skip_ligand_prep:\n")
                    iyf.write(' - "' + ligand_pdb_name[ligand] + '"\n')

                iyf.write("box_radius: " + str(box_radius) + "\n")
                if isinstance(box_centers, type(None)) and peptide:
                    raise ValueError(
                        "You must give per-protein box_centers when docking peptides!"
                    )
                if not isinstance(box_centers, type(None)):
                    if (
                        not all(
                            isinstance(x, float) for x in box_centers[model]
                        )
                        and not all(
                            isinstance(x, int) for x in box_centers[model]
                        )
                        and not all(
                            isinstance(x, np.float32)
                            for x in box_centers[model]
                        )
                    ):
                        # get coordinates from tuple
                        coordinates = None
                        for chain in self.structures[model[0]].get_chains():
                            if chain.id == box_centers[model][0]:
                                for r in chain:
                                    if r.id[1] == box_centers[model][1]:
                                        for atom in r:
                                            if (
                                                atom.name
                                                == box_centers[model][2]
                                            ):
                                                coordinates = atom.coord
                        if isinstance(coordinates, type(None)):
                            raise ValueError(
                                f"Atom {box_centers[model]} was not found for protein {model[0]}"
                            )
                    else:
                        coordinates = box_centers[model]

                    box_center = ""
                    for coord in coordinates:
                        # if not isinstance(coord, float):
                        #    raise ValueError('Box centers must be given as a (x,y,z) tuple or list of floats.')
                        box_center += "  - " + str(float(coord)) + "\n"
                    iyf.write("box_center: \n" + box_center)

                # energy by residue is not implemented in PELE platform, therefore
                # a scond script will modify the PELE.conf file to set up the energy
                # by residue calculation.
                if any(
                    [
                        debug,
                        energy_by_residue,
                        peptide,
                        nonbonded_energy != None,
                        membrane_residues,
                        bias_to_point,
                        com_bias1,
                        com_residue_pairs,
                        ligand_equilibration_cst,
                        regional_spawning,
                        constraint_level,
                    ]
                ):
                    iyf.write("debug: true\n")

                if distances != None:
                    iyf.write("atom_dist:\n")
                    for d in distances[protein][ligand]:
                        if isinstance(d[0], str):
                            d1 = (
                                "- 'L:" + str(ligand_index) + ":" + d[0] + "'\n"
                            )
                        else:
                            d1 = (
                                "- '"
                                + d[0][0]
                                + ":"
                                + str(d[0][1])
                                + ":"
                                + d[0][2]
                                + "'\n"
                            )
                        if isinstance(d[1], str):
                            d2 = (
                                "- 'L:" + str(ligand_index) + ":" + d[1] + "'\n"
                            )
                        else:
                            d2 = (
                                "- '"
                                + d[1][0]
                                + ":"
                                + str(d[1][1])
                                + ":"
                                + d[1][2]
                                + "'\n"
                            )
                        iyf.write(d1)
                        iyf.write(d2)

                if constraints != None:
                    iyf.write("external_constraints:\n")
                    for c in constraints[(protein, ligand)]:
                        if len(c) == 2:
                            line = (
                                "- '"
                                + str(c[0])
                                + "-"
                                + str(c[1][0])
                                + ":"
                                + str(c[1][1])
                                + ":"
                                + str(c[1][2])
                                + "'\n"
                            )  # cst_force and atom_index for positional cst
                        elif len(c) == 4:
                            line = (
                                "- '"
                                + str(c[0])
                                + "-"
                                + str(c[1])
                                + "-"
                                + str(c[2][0])
                                + ":"
                                + str(c[2][1])
                                + ":"
                                + str(c[2][2])
                                + "-"
                                + str(c[3][0])
                                + ":"
                                + str(c[3][1])
                                + ":"
                                + str(c[3][2])
                                + "'\n"
                            )  # cst_force, distance, atom_index1, atom_index2 for distance cst
                        else:
                            raise ValueError(
                                "Constraint for protein "
                                + protein
                                + " with ligand "
                                + ligand
                                + " are not defined correctly."
                            )
                        iyf.write(line)

                if seed:
                    iyf.write("seed: " + str(seed) + "\n")

                if log_file:
                    iyf.write("log: true\n")

                iyf.write("\n")
                iyf.write("#Options gathered from " + input_yaml + "\n")

                with open(input_yaml) as tyf:
                    for l in tyf:
                        if l.startswith("#"):
                            continue
                        elif l.startswith("-"):
                            continue
                        elif l.strip() == "":
                            continue
                        if l.split()[0].replace(":", "") not in keywords:
                            iyf.write(l)

            if energy_by_residue:
                _copyScriptFile(pele_folder, "addEnergyByResidueToPELEconf.py")
                ebr_script_name = "._addEnergyByResidueToPELEconf.py"
                if not isinstance(ligand_energy_groups, type(None)):
                    if not isinstance(ligand_energy_groups, dict):
                        raise ValueError(
                            "ligand_energy_groups, must be given as a dictionary"
                        )
                    with open(
                        protein_ligand_folder + "/ligand_energy_groups.json",
                        "w",
                    ) as jf:
                        json.dump(ligand_energy_groups[ligand], jf)

            if protein in membrane_residues:
                _copyScriptFile(pele_folder, "addMembraneConstraints.py")
                mem_res_script = (
                    "._addMembraneConstraints.py"  # I have added the _
                )

            if nonbonded_energy != None:
                _copyScriptFile(
                    pele_folder, "addAtomNonBondedEnergyToPELEconf.py"
                )
                nbe_script_name = "._addAtomNonBondedEnergyToPELEconf.py"
                if not isinstance(nonbonded_energy, dict):
                    raise ValueError(
                        "nonbonded_energy, must be given as a dictionary"
                    )
                with open(
                    protein_ligand_folder + "/nonbonded_energy_atoms.json", "w"
                ) as jf:
                    json.dump(nonbonded_energy[protein][ligand], jf)

            if protein in bias_to_point:
                _copyScriptFile(pele_folder, "addBiasToPoint.py")
                btp_script = "._addBiasToPoint.py"

            if protein in com_bias1:
                _copyScriptFile(pele_folder, "addComDistancesBias.py")
                cbs_script = "._addComDistancesBias.py"

            if protein in com_residue_pairs:
                _copyScriptFile(pele_folder, "addComDistances.py")
                cds_script = "._addComDistances.py"

            if peptide:
                _copyScriptFile(pele_folder, "modifyPelePlatformForPeptide.py")
                peptide_script_name = "._modifyPelePlatformForPeptide.py"

            if ligand_equilibration_cst:
                _copyScriptFile(
                    pele_folder, "addLigandConstraintsToPELEconf.py"
                )
                equilibration_script_name = (
                    "._addLigandConstraintsToPELEconf.py"
                )
                _copyScriptFile(pele_folder, "changeAdaptiveIterations.py")
                adaptive_script_name = "._changeAdaptiveIterations.py"

            if restore_input_coordinates:
                _copyScriptFile(
                    pele_folder, 'restoreChangedCoordinates.py'
                )
                restore_coordinates_script_name = "._restoreChangedCoordinates.py"

            # Create command
            command = "cd " + protein_ligand_folder + "\n"

            # Add commands to write template folder absolute paths
            if ligand in templates:
                command += "export CWD=$(pwd)\n"
                command += "cd ../templates/" + ligand + "\n"
                command += "export TMPLT_DIR=$(pwd)\n"
                command += "cd $CWD\n"
                for tf in templates[ligand]:
                    if continuation:
                        yaml_file = "input_restart.yaml"
                    else:
                        yaml_file = "input.yaml"
                    if tf.endswith(".assign"):
                        command += (
                            "sed -i s,LIGAND_TEMPLATE_PATH_ROT,$TMPLT_DIR/"
                            + tf
                            + ",g "
                            + yaml_file
                            + "\n"
                        )
                    elif tf.endswith("z"):
                        command += (
                            "sed -i s,LIGAND_TEMPLATE_PATH_Z,$TMPLT_DIR/"
                            + tf
                            + ",g "
                            + yaml_file
                            + "\n"
                        )

            if not continuation:
                command += "python -m pele_platform.main input.yaml\n"

                if regional_spawning:
                    continuation = True

                if angles:
                    # Copy individual angle definitions to each protein and ligand folder
                    if protein in angles and ligand in angles[protein]:
                        with open(
                            protein_ligand_folder + "/._angles.json", "w"
                        ) as jf:
                            json.dump(angles[protein][ligand], jf)

                    # Copy script to add angles to pele.conf
                    _copyScriptFile(pele_folder, "addAnglesToPELEConf.py")
                    command += (
                        "python "
                        + rel_path_to_root
                        + "._addAnglesToPELEConf.py output "
                    )
                    command += "._angles.json "
                    command += (
                        "output/input/"
                        + protein_ligand
                        + separator
                        + pose
                        + "_processed.pdb\n"
                    )
                    continuation = True

                if constraint_level:
                    # Copy script to add angles to pele.conf
                    _copyScriptFile(
                        pele_folder, "correctPositionalConstraints.py"
                    )
                    command += (
                        "python "
                        + rel_path_to_root
                        + "._correctPositionalConstraints.py output "
                    )
                    command += (
                        "output/input/"
                        + protein_ligand
                        + separator
                        + pose
                        + "_processed.pdb\n"
                    )
                    continuation = True

                if energy_by_residue:
                    command += (
                        "python "
                        + rel_path_to_root
                        + ebr_script_name
                        + " output --energy_type "
                        + energy_by_residue_type
                        + "--new_version "
                        + new_version
                    )
                    if isinstance(ligand_energy_groups, dict):
                        command += (
                            " --ligand_energy_groups ligand_energy_groups.json"
                        )
                        command += " --ligand_index " + str(ligand_index)
                    if ebr_new_flag:
                        command += " --new_version "
                    if peptide:
                        command += " --peptide \n"
                        command += (
                            "python "
                            + rel_path_to_root
                            + peptide_script_name
                            + " output "
                            + " ".join(models[model])
                            + "\n"
                        )
                    else:
                        command += "\n"

                if protein in membrane_residues:
                    command += (
                        "python " + rel_path_to_root + mem_res_script + " "
                    )
                    command += "output "  # I think we should change this for a variable
                    command += "--membrane_residues "
                    command += (
                        ",".join([str(x) for x in membrane_residues[protein]])
                        + "\n"
                    )  # 1,2,3,4,5
                    continuation = True

                if protein in bias_to_point:
                    command += "python " + rel_path_to_root + btp_script + " "
                    command += "output "  # I think we should change this for a variable
                    command += (
                        "point_"
                        + ",".join([str(x) for x in bias_to_point[protein]])
                        + " "
                    )
                    command += "--epsilon " + str(epsilon) + "\n"
                    continuation = True

                if protein in com_bias1 and ligand in com_bias1[protein]:
                    # Write both COM groups as json files
                    with open(
                        protein_ligand_folder + "/._com_group1.json", "w"
                    ) as jf:
                        json.dump(com_bias1[protein][ligand], jf)

                    with open(
                        protein_ligand_folder + "/._com_group2.json", "w"
                    ) as jf:
                        json.dump(com_bias2[protein][ligand], jf)

                    command += "python " + rel_path_to_root + cbs_script + " "
                    command += "output "  # I think we should change this for a variable
                    command += "._com_group1.json "
                    command += "._com_group2.json "
                    command += "--epsilon " + str(epsilon) + "\n"
                    continuation = True

                if protein in com_residue_pairs and ligand in com_residue_pairs[protein]:
                    # Write COM groups as json file
                    with open(
                        protein_ligand_folder + "/._com_groups.json", "w"
                    ) as jf:
                        json.dump(com_residue_pairs[protein][ligand], jf)

                    command += "python " + rel_path_to_root + cds_script + " "
                    command += "output "  # I think we should change this for a variable
                    command += "._com_groups.json\n"
                    continuation = True

                if covalent_setup:
                    command += covalent_command
                    continuation = True

                if restore_input_coordinates:
                    command += "python "+ rel_path_to_root+restore_coordinates_script_name+" "
                    command += "output/input/"+protein_ligand+separator+pose+".pdb "
                    command += "output/input/"+protein_ligand+separator+pose+"_processed.pdb\n"
                    continuation = True

                if ligand_equilibration_cst:

                    # Copy input_yaml for equilibration
                    oyml = open(
                        protein_ligand_folder + "/input_equilibration.yaml", "w"
                    )
                    debug_line = False
                    restart_line = False
                    with open(protein_ligand_folder + "/input.yaml") as iyml:
                        for l in iyml:
                            if "debug: true" in l:
                                debug_line = True
                                oyml.write("restart: true\n")
                                oyml.write("adaptive_restart: true\n")
                                continue
                            elif "restart: true" in l:
                                restart_line = True
                            elif l.startswith("iterations:"):
                                l = "iterations: 1\n"
                            elif l.startswith("steps:"):
                                l = "steps: 1\n"
                            oyml.write(l)
                        if not debug_line and not restart_line:
                            oyml.write("restart: true\n")
                            oyml.write("adaptive_restart: true\n")
                    oyml.close()

                    # Add commands for adding ligand constraints
                    command += "cp output/pele.conf output/pele.conf.backup\n"
                    command += (
                        "cp output/adaptive.conf output/adaptive.conf.backup\n"
                    )

                    # Modify pele.conf to add ligand constraints
                    command += (
                        "python "
                        + rel_path_to_root
                        + equilibration_script_name
                        + " "
                    )
                    command += "output "  # I think we should change this for a variable
                    if (
                        isinstance(ligand_equilibration_cst, (int, float))
                        and ligand_equilibration_cst != 1.0
                    ):
                        command += "--constraint_value " + str(
                            float(ligand_equilibration_cst)
                        )
                    command += "\n"

                    # Modify adaptive.conf to remove simulation steps
                    command += (
                        "python "
                        + rel_path_to_root
                        + adaptive_script_name
                        + " "
                    )
                    command += "output "  # I think we should change this for a variable
                    command += "--iterations 1 "
                    command += "--steps 1\n"

                    # Launch equilibration
                    command += "python -m pele_platform.main input_equilibration.yaml\n"

                    # Recover conf files
                    command += "cp output/pele.conf.backup output/pele.conf\n"
                    command += (
                        "cp output/adaptive.conf.backup output/adaptive.conf\n"
                    )
                    continuation = True

            if continuation:
                debug_line = False
                restart_line = False
                # Copy input_yaml for equilibration
                oyml = open(protein_ligand_folder + "/input_restart.yaml", "w")
                debug_line = False
                restart_line = False
                with open(protein_ligand_folder + "/input.yaml") as iyml:
                    for l in iyml:
                        if "debug: true" in l:
                            debug_line = True
                            oyml.write("restart: true\n")
                            oyml.write("adaptive_restart: true\n")
                            continue
                        elif "restart: true" in l:
                            restart_line = True
                        oyml.write(l)
                    if not debug_line and not restart_line:
                        oyml.write("restart: true\n")
                        oyml.write("adaptive_restart: true\n")
                oyml.close()

                if extend_iterations:
                    _copyScriptFile(pele_folder, "changeAdaptiveIterations.py")
                    extend_script_name = "._changeAdaptiveIterations.py"
                    command += (
                        "python "
                        + rel_path_to_root
                        + extend_script_name
                        + " output "  # I think we should change this for a variable
                        + "--iterations "
                        + str(iterations)+' '
                        + "--steps "
                        + str(steps)+' '
                        + "\n"
                    )
                if not energy_by_residue:
                    command += (
                        "python -m pele_platform.main input_restart.yaml\n"
                    )

                if (
                    any(
                        [
                            membrane_residues,
                            bias_to_point,
                            com_bias1,
                            ligand_equilibration_cst,
                            angles,
                            regional_spawning,
                            constraint_level,
                        ]
                    )
                    and not continue_all
                ):
                    continuation = False
                    debug = False

            elif peptide:
                command += (
                    "python "
                    + rel_path_to_root
                    + peptide_script_name
                    + " output "
                    + " ".join(models[model])
                    + "\n"
                )
                with open(
                    protein_ligand_folder + "/" + "input_restart.yaml", "w"
                ) as oyml:
                    with open(
                        protein_ligand_folder + "/" + "input.yaml"
                    ) as iyml:
                        for l in iyml:
                            if "debug: true" in l:
                                l = "restart: true\n"
                            oyml.write(l)
                if nonbonded_energy == None:
                    command += (
                        "python -m pele_platform.main input_restart.yaml\n"
                    )

            elif extend_iterations and not continuation:
                raise ValueError(
                    "extend_iterations must be used together with the continuation keyword"
                )

            if nonbonded_energy != None:
                command += (
                    "python "
                    + rel_path_to_root
                    + nbe_script_name
                    + " output --energy_type "
                    + nonbonded_energy_type
                )
                command += " --target_atoms nonbonded_energy_atoms.json"
                protein_chain = [
                    c for c in self.structures[protein].get_chains() if c != "L"
                ][0]
                command += " --protein_chain " + protein_chain.id
                if ebr_new_flag or nonbonded_new_flag:
                    command += " --new_version"
                command += "\n"

                if not os.path.exists(
                    protein_ligand_folder + "/" + "input_restart.yaml"
                ):
                    with open(
                        protein_ligand_folder + "/" + "input_restart.yaml", "w"
                    ) as oyml:
                        with open(
                            protein_ligand_folder + "/" + "input.yaml"
                        ) as iyml:
                            for l in iyml:
                                if "debug: true" in l:
                                    l = "restart: true\n"
                                oyml.write(l)
                command += "python -m pele_platform.main input_restart.yaml\n"

            # Remove debug line from input.yaml for covalent setup (otherwise the Data folder is not copied!)
            if covalent_setup:
                with open(
                    protein_ligand_folder + "/" + "input.yaml.tmp", "w"
                ) as oyf:
                    with open(
                        protein_ligand_folder + "/" + "input.yaml"
                    ) as iyf:
                        for l in iyf:
                            if not "debug: true" in l:
                                oyf.write(l)
                shutil.move(
                    protein_ligand_folder + "/" + "input.yaml.tmp",
                    protein_ligand_folder + "/" + "input.yaml",
                )

            if regional_spawning:
                command += "cd ../\n"
                command += "python ../._regionalSpawning.py "
                command += "metrics.json "
                command += "metrics_thresholds.json "
                if regional_combinations:
                    command += "--combinations regional_combinations.json "
                    command += "--exclusions regional_exclusions.json "
                command += "--separator " + separator + " "
                command += '--energy_bias "' + regional_energy_bias + '" '
                command += (
                    "--regional_best_fraction "
                    + str(regional_best_fraction)
                    + " "
                )
                if max_regional_iterations:
                    command += (
                        "--max_iterations " + str(max_regional_iterations) + " "
                    )
                if angles:
                    command += "--angles "
                if restore_input_coordinates:
                    command += '--restore_coordinates '
                command += "\n"

            command += "cd ../../"
            jobs.append(command)

        return jobs

    def setUpMDSimulations(self, md_folder, sim_time, nvt_time=2, npt_time=0.2, equilibration_dt=2, production_dt=2,
                           temperature=298.15, frags=1, local_command_name=None, remote_command_name="${GMXBIN}",
                           ff="amber99sb-star-ildn", ligand_chains=None, ion_chains=None, replicas=1, charge=None,
                           system_output="System", models=None, overwrite=False, remove_backups=False,constantph=False):
        """
        Sets up MD simulations for each model. The current state only allows to set
        up simulations using the Gromacs software.

        If the input pdb has additional non aa residues besides ligand (ions, HETATMs, ...)
        they should be separated in individual chains.

        Parameters:
        ==========
        md_folder : str
            Path to the job folder where the MD input files are located.
        sim_time : int
            Simulation time in ns
        nvt_time : float
            Time for NVT equilibration in ns
        npt_time : float
            Time for NPT equilibration in ns
        equilibration_dt : float
            Time step for equilibration in fs
        production_dt : float
            Time step for production in fs
        temperature : float
            Simulation temperature in K
        frags : int
            Number of fragments to divide the simulation.
        local_command_name : str
            Local command name for Gromacs.
        remote_command_name : str
            Remote command name for Gromacs.
        ff : str
            Force field to use for simulation.
        ligand_chains : list
            List of ligand chains.
        ion_chains : list
            List of ion chains.
        replicas : int
            Number of replicas.
        charge : int
            Charge of the system.
        system_output : str
            Output system name.
        models : list
            List of models.
        overwrite : bool
            Whether to overwrite existing files.
        remove_backups : bool
            Whether to remove backup files generated by Gromacs.
        """

        def _copyScriptFile(dest_folder, file, subfolder=None):
            source = resource_stream(Requirement.parse("prepare_proteins"), f"prepare_proteins/scripts/{subfolder}/{file}")
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            with open(os.path.join(dest_folder, file), 'wb') as dest_file:
                shutil.copyfileobj(source, dest_file)

        def _readGromacsIndexFile(file):
            with open(file, 'r') as f:
                groups = [x.replace('[', '').replace(']', '').replace('\n', '').strip() for x in f.readlines() if x.startswith('[')]
            return {g: str(i) for i, g in enumerate(groups)}

        def _getLigandParameters(structure, ligand_chains, struct_path, params_path, charge=None):
            class chainSelect(PDB.Select):
                def accept_chain(self, chain):
                    return chain.get_id() not in ligand_chains

            charge = charge or {}
            ligand_res = {chain.get_id(): residue.resname for mdl in structure for chain in mdl for residue in chain if chain.get_id() in ligand_chains}
            if not ligand_res:
                raise ValueError(f"Ligand was not found at chains {str(ligand_chains)}")

            io = PDB.PDBIO()
            pdb_chains = list(structure.get_chains())
            if len(pdb_chains) < 2:
                raise ValueError("Input pdb has only one chain. Protein and ligand should be separated in individual chains.")

            io.set_structure(structure)
            io.save(f"{struct_path}/protein.pdb", chainSelect())

            ligand_coords = {chain.get_id(): [a.coord for a in chain.get_atoms()] for chain in pdb_chains if chain.get_id() in ligand_chains}
            for chain_id, coords in ligand_coords.items():
                io.set_structure([chain for chain in pdb_chains if chain.get_id() == chain_id][0])
                io.save(f"{struct_path}/{ligand_res[chain_id]}.pdb")

            lig_counter = 0
            for lig_chain, ligand_name in ligand_res.items():
                lig_counter += 1
                if ligand_name not in os.listdir(params_path) or overwrite:
                    os.makedirs(f"{params_path}/{ligand_name}", exist_ok=True)
                    shutil.copyfile(f"{struct_path}/{ligand_name}.pdb", f"{params_path}/{ligand_name}/{ligand_name}.pdb")
                    os.chdir(f"{params_path}/{ligand_name}")

                    command = f"acpype -i {ligand_name}.pdb"
                    if ligand_name in charge:
                        command += f" -n {charge[ligand_name]}"
                    subprocess.run(command, shell=True)

                    with open(f"{ligand_name}.acpype/{ligand_name}_GMX.itp") as f:
                        lines = f.readlines()

                    atomtypes_lines, new_lines = [], []
                    atomtypes, atoms = False, False
                    for i, line in enumerate(lines):
                        if atomtypes:
                            if line.startswith("[ moleculetype ]"):
                                new_lines.append(line)
                                atomtypes = False
                            else:
                                spl = line.split()
                                if spl:
                                    spl[0] = ligand_name + spl[0]
                                    spl[1] = ligand_name + spl[1]
                                    atomtypes_lines.append(" ".join(spl))
                        elif atoms:
                            if line.startswith("[ bonds ]"):
                                new_lines.append(line)
                                atoms = False
                            else:
                                spl = line.split()
                                if spl:
                                    spl[1] = ligand_name + spl[1]
                                    new_lines.append(" ".join(spl) + "\n")
                        else:
                            new_lines.append(line)

                        if line.startswith(";name"):
                            if lines[i - 1].startswith("[ atomtypes ]"):
                                atomtypes = True

                        elif line.startswith(";"):
                            if lines[i - 1].startswith("[ atoms ]"):
                                atoms = True

                    print(lig_counter)
                    write_type = "w" if lig_counter == 1 else "a"
                    with open("../atomtypes.itp", write_type) as f:
                        if lig_counter == 1:
                            f.write("[ atomtypes ]\n")
                        for line in atomtypes_lines:
                            f.write(line + "\n")

                    with open(f"{ligand_name}.acpype/{ligand_name}_GMX.itp", "w") as f:
                        for line in new_lines:
                            if not line.startswith("[ atomtypes ]"):
                                f.write(line)

                    os.chdir("../../..")

                parser = PDB.PDBParser()
                ligand_structure = parser.get_structure("ligand", f"{params_path}/{ligand_name}/{ligand_name}.acpype/{ligand_name}_NEW.pdb")
                for i, atom in enumerate(ligand_structure.get_atoms()):
                    atom.coord = ligand_coords[lig_chain][i]
                io.set_structure(ligand_structure)
                io.save(f"{struct_path}/{ligand_name}.pdb")

            return ligand_res

        def _setupModelStructure(structure, ligand_chains, ion_chains):
            gmx_codes, ion_residues = [], []
            for mdl in structure:
                for chain in mdl:
                    for residue in chain:
                        if ion_chains and chain.get_id() in ion_chains:
                            ion_residues.append(residue.id[1])
                        HD1, HE2 = False, False
                        if residue.resname == "HIS":
                            for atom in residue:
                                if atom.name == "HD1":
                                    HD1 = True
                                if atom.name == "HE2":
                                    HE2 = True
                        if HD1 or HE2:
                            number = 0 if HD1 and not HE2 else 1 if HE2 and not HD1 else 2
                            gmx_codes.append(number)
            return str(gmx_codes)[1:-1].replace(",", ""), ion_residues

        def _createCAConstraintFile(structure, cst_file, sd=1.0):
            with open(cst_file, "w") as f:
                ref_res, ref_chain = None, None
                for r in structure.get_residues():
                    if r.id[0] != " ":
                        continue
                    res, chain = r.id[1], r.get_parent().id
                    if not ref_res:
                        ref_res, ref_chain = res, chain
                    if ref_chain != chain:
                        ref_res, ref_chain = res, chain

                    ca_atom = None
                    for atom in r.get_atoms():
                        if atom.name == "CA":
                            ca_atom = atom

                    if ca_atom != None:
                        ca_coordinate = list(ca_atom.coord)
                        cst_line = f"CoordinateConstraint CA {res}{chain} CA {ref_res}{ref_chain} "
                        cst_line += " ".join([f"{c:.4f}" for c in ca_coordinate]) + f" HARMONIC 0 {sd}\n"
                        f.write(cst_line)

        def _generateLocalCommand(command_name, model, i, ligand_chains, ligand_res, ion_residues, ion_chains, md_folder, ff, his_pro):
            command_local = f"cd {md_folder}\nexport GMXLIB=$(pwd)/FF\nmkdir -p output_models/{model}/{i}/topol\n"
            command_local += f"cp input_models/{model}/protein.pdb output_models/{model}/{i}/topol/protein.pdb\n"

            if ligand_chains:
                command_local += f"cp ligand_params/atomtypes.itp output_models/{model}/{i}/topol/atomtypes.itp\n"
                for ligand_name in ligand_res.values():
                    command_local += f"cp -r ligand_params/{ligand_name}/{ligand_name}.acpype output_models/{model}/{i}/topol/\n"

            command_local += f"cd output_models/{model}/{i}/topol\n"
            command_local += f"echo {his_pro} | {command_name} pdb2gmx -f protein.pdb -o prot.pdb -p topol.top -his -ignh -ff {ff} -water tip3p -vsite hydrogens\n"

            if ligand_chains:
                lig_files = " ".join([f" ../../../../input_models/{model}/{ligand_name}.pdb " for ligand_name in ligand_res.values()])
                command_local += f"grep -h ATOM prot.pdb {lig_files} >| complex.pdb\n"
                command_local += f"{command_name} editconf -f complex.pdb -o complex.gro\n"
                line = '#include "atomtypes.itp"\\n'
                included_ligands = []
                for ligand_name in ligand_res.values():
                    if ligand_name not in included_ligands:
                        included_ligands.append(ligand_name)
                        line += f'#include "{ligand_name}.acpype/{ligand_name}_GMX.itp"\\n'
                    line += "#ifdef POSRES\\n"
                    line += f'#include "{ligand_name}.acpype/posre_{ligand_name}.itp"\\n'
                    line += "#endif\\n"
                line += "'"
                local_path = (os.getcwd() + "/" + md_folder + "/FF").replace("/", "\/")
                command_local += f"sed -i '/#include \"{local_path}\/{ff}.ff\/forcefield.itp\"/a {line} topol.top\n"
                for ligand_name in ligand_res.values():
                    command_local += f"sed -i -e '$a{ligand_name.ljust(20)}1' topol.top\n"
            else:
                command_local += f"{command_name} editconf -f prot.pdb -o complex.gro\n"

            command_local += f"{command_name} editconf -f complex.gro -o prot_box.gro -c -d 1.0 -bt octahedron\n"
            command_local += f"{command_name} solvate -cp prot_box.gro -cs spc216.gro -o prot_solv.gro -p topol.top\n"
            command_local += f'echo q | {command_name} make_ndx -f prot_solv.gro -o index.ndx\n'

            return command_local

        def _removeBackupFiles(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.startswith("#") and file.endswith("#"):
                        os.remove(os.path.join(root, file))

        remote_command_name = "${GMXBIN}"
        if isinstance(models, str):
            models = [models]
        if not os.path.exists(md_folder):
            os.mkdir(md_folder)
        if not os.path.exists(f"{md_folder}/scripts"):
            os.mkdir(f"{md_folder}/scripts")
        if not os.path.exists(f"{md_folder}/FF"):
            os.mkdir(f"{md_folder}/FF")
        if not os.path.exists(f"{md_folder}/FF/{ff}.ff"):
            os.mkdir(f"{md_folder}/FF/{ff}.ff")
        if not os.path.exists(f"{md_folder}/input_models"):
            os.mkdir(f"{md_folder}/input_models")
        if not os.path.exists(f"{md_folder}/output_models"):
            os.mkdir(f"{md_folder}/output_models")

        if local_command_name is None:
            possible_command_names = ["gmx", "gmx_mpi"]
            command_name = None
            for command in possible_command_names:
                if shutil.which(command) is not None:
                    command_name = command
                    break
            if command_name is None:
                raise ValueError(f"Gromacs executable is required for the setup and was not found. The following executable names were tested: {','.join(possible_command_names)}")
        else:
            command_name = local_command_name

        if ligand_chains is not None:
            if isinstance(ligand_chains, str):
                ligand_chains = [ligand_chains]
            if not os.path.exists(f"{md_folder}/ligand_params"):
                os.mkdir(f"{md_folder}/ligand_params")

        self.saveModels(f"{md_folder}/input_models")

        for file in resource_listdir(Requirement.parse("prepare_proteins"), "prepare_proteins/scripts/md/gromacs/mdp"):
            if not file.startswith("__"):
                _copyScriptFile(f"{md_folder}/scripts", file, subfolder="md/gromacs/mdp")

        for file in resource_listdir(Requirement.parse("prepare_proteins"), f"prepare_proteins/scripts/md/gromacs/ff/{ff}"):
            if not file.startswith("__"):
                _copyScriptFile(f"{md_folder}/FF/{ff}.ff", file, subfolder=f"md/gromacs/ff/{ff}")

        for line in fileinput.input(f"{md_folder}/scripts/em.mdp", inplace=True):
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        for line in fileinput.input(f"{md_folder}/scripts/md.mdp", inplace=True):
            if "TIME_INTEGRATOR" in line:
                line = line.replace("TIME_INTEGRATOR", str(production_dt / 1000))
                if equilibration_dt > 2:
                    print("WARNING: you have selected a time integrator higher than 2 femtoseconds. Constraints have been automatically changed to all bonds. This may affect the accuracy of your simulation.")
                    cst = "all-bonds"
                else:
                    cst = "h-bonds"
            if "BOND_CONSTRAINTS" in line:
                line = line.replace("BOND_CONSTRAINTS", cst)
            if "NUMBER_OF_STEPS" in line:
                line = line.replace("NUMBER_OF_STEPS", str(int((sim_time * (1e6 / production_dt)) / frags)))
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        for line in fileinput.input(f"{md_folder}/scripts/nvt.mdp", inplace=True):
            if "TIME_INTEGRATOR" in line:
                line = line.replace("TIME_INTEGRATOR", str(equilibration_dt / 1000))
                if equilibration_dt > 2:
                    print("WARNING: you have selected a time integrator higher than 2 femtoseconds. Constraints have been automatically changed to all bonds. This may affect the accuracy of your simulation.")
                    cst = "all-bonds"
                else:
                    cst = "h-bonds"
            if "BOND_CONSTRAINTS" in line:
                line = line.replace("BOND_CONSTRAINTS", cst)
            if "NUMBER_OF_STEPS" in line:
                line = line.replace("NUMBER_OF_STEPS", str(int(nvt_time * (1e6 / equilibration_dt))))
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        for line in fileinput.input(f"{md_folder}/scripts/npt.mdp", inplace=True):
            if "TIME_INTEGRATOR" in line:
                line = line.replace("TIME_INTEGRATOR", str(equilibration_dt / 1000))
                if equilibration_dt > 2:
                    print("WARNING: you have selected a time integrator higher than 2 femtoseconds. Constraints have been automatically changed to all bonds. This may affect the accuracy of your simulation.")
                    cst = "all-bonds"
                else:
                    cst = "h-bonds"
            if "BOND_CONSTRAINTS" in line:
                line = line.replace("BOND_CONSTRAINTS", cst)
            if "NUMBER_OF_STEPS" in line:
                line = line.replace("NUMBER_OF_STEPS", str(int(npt_time * (1e6 / equilibration_dt))))
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        jobs = []
        for model in self.models_names:
            if models and model not in models:
                continue

            if not os.path.exists(f"{md_folder}/input_models/{model}"):
                os.mkdir(f"{md_folder}/input_models/{model}")
            if not os.path.exists(f"{md_folder}/output_models/{model}"):
                os.mkdir(f"{md_folder}/output_models/{model}")

            for i in range(replicas):
                if not os.path.exists(f"{md_folder}/output_models/{model}/{i}"):
                    os.mkdir(f"{md_folder}/output_models/{model}/{i}")

            parser = PDB.PDBParser()
            structure = parser.get_structure("protein", f"{md_folder}/input_models/{model}.pdb")

            his_pro, ion_residues = _setupModelStructure(structure, ligand_chains, ion_chains)
            ligand_res = _getLigandParameters(structure, ligand_chains, f"{md_folder}/input_models/{model}", f"{md_folder}/ligand_params", charge=charge) if ligand_chains else shutil.copyfile(f"{md_folder}/input_models/{model}.pdb",f"{md_folder}/input_models/{model}/protein.pdb")

            for i in range(replicas):

                skip_local = os.path.exists(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx") and not overwrite
                if not skip_local:
                    command_local = _generateLocalCommand(command_name, model, i, ligand_chains, ligand_res, ion_residues, ion_chains, md_folder, ff, his_pro)
                    print(command_local)
                    #adfasdf
                    with open("tmp.sh", "w") as f:
                        f.write(command_local)
                    subprocess.run("bash tmp.sh", shell=True)
                    os.remove("tmp.sh")

                group_dics = {}
                group_dics["complex"] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx")

                if not skip_local:
                    os.system(f'echo "q"| {command_name} make_ndx -f {md_folder}/output_models/{model}/{i}/topol/complex.gro -o {md_folder}/output_models/{model}/{i}/topol/tmp_index.ndx')
                    group_dics['tmp_index'] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/tmp_index.ndx")

                    if 'Water' in group_dics['tmp_index']:
                        reading, crystal_waters_ndx_lines = False, '[ CrystalWaters ]\n'
                        for line in open(f"{md_folder}/output_models/{model}/{i}/topol/tmp_index.ndx"):
                            if '[' in line and reading:
                                reading = False
                            elif '[ Water ]' in line:
                                reading = True
                            elif reading:
                                crystal_waters_ndx_lines += line

                        with open(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx", 'a') as f:
                            f.write(crystal_waters_ndx_lines)
                        os.system(f'echo \"{group_dics["complex"]["Water"]} & !{len(group_dics["complex"])}\nq\" | {command_name} make_ndx -f {md_folder}/output_models/{model}/{i}/topol/prot_solv.gro -o {md_folder}/output_models/{model}/{i}/topol/index.ndx -n {md_folder}/output_models/{model}/{i}/topol/index.ndx')
                        os.system(f'echo \"del {group_dics["complex"]["SOL"]}\n name {len(group_dics["complex"])} SOL\nq\" | {command_name} make_ndx -f {md_folder}/output_models/{model}/{i}/topol/prot_solv.gro -o {md_folder}/output_models/{model}/{i}/topol/index.ndx -n {md_folder}/output_models/{model}/{i}/topol/index.ndx')

                        group_dics['complex'] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx")

                sol_group = 'SOL'
                skip_ions = os.path.exists(f"{md_folder}/output_models/{model}/{i}/topol/prot_ions.gro") and not overwrite
                if not skip_ions:
                    command_local = f"cd {md_folder}/output_models/{model}/{i}/topol\n"
                    command_local += f"{command_name} grompp -f ../../../../scripts/ions.mdp -c prot_solv.gro -p topol.top -o prot_ions.tpr -maxwarn 1\n"
                    command_local += f"echo {group_dics['complex'][sol_group]} | {command_name} genion -s prot_ions.tpr -o prot_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.1 -n index.ndx\n"
                    # if constatph add buffer
                    if constantph:
                        command_local += f"{command_name} grompp -f ../../../../scripts/ions.mdp -c prot_ions.gro -p topol.top -o prot_buf.tpr -maxwarn 1\n"
                        command_local += f"echo {group_dics['complex'][sol_group]} | {command_name} genion -s prot_buf.tpr -p topol.top -o prot_buf.gro -np 1 -rmin 1.0 -pname BUF -n index.ndx\n"
                        file_name = 'prot_buf'
                    else:
                        file_name = 'prot_ions'

                    command_local += f'echo "q"| {command_name} make_ndx -f {file_name}.gro -o index.ndx\n'

                    with open("tmp.sh", "w") as f:
                        f.write(command_local)
                    subprocess.run("bash tmp.sh", shell=True)
                    os.remove("tmp.sh")

                    group_dics["complex"] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx")

                if ligand_chains or ion_residues:
                    skip_ndx = os.path.exists(f"{md_folder}/output_models/{model}/{i}/topol/posre.itp") and not overwrite
                    if not skip_ndx:
                        command_local = f"cd {md_folder}/output_models/{model}/{i}/topol\n"
                        lig_selector = ""
                        if ligand_chains:
                            for ligand_name in ligand_res.values():
                                command_local += f'echo -e "0 & ! a H*\nq"| {command_name} make_ndx -f {ligand_name}.acpype/{ligand_name}_GMX.gro -o {ligand_name}_index.ndx\n'
                                lig_selector += f"{group_dics['complex'][ligand_name]}|"

                        ion_selector, water_and_solventions_selector = "", ""
                        if ion_residues:
                            for r in ion_residues:
                                ion_selector += f"r {r}|"
                                water_and_solventions_selector += f" ! r {r} &"

                        selector_line = ""
                        if lig_selector and ion_selector:
                            selector_line += f"{group_dics['complex']['Protein']}|{ion_selector[:-1]}|{lig_selector[:-1]}\n"
                            selector_line += f"{group_dics['complex']['Protein']}|{ion_selector[:-1]}\n"
                            selector_line += f"{group_dics['complex'][sol_group]} | {group_dics['complex']['Ion']} & {water_and_solventions_selector[:-1]}\n"
                        elif ion_selector:
                            selector_line += f"{group_dics['complex']['Protein']}|{ion_selector[:-1]}\n"
                            selector_line += f"{group_dics['complex'][sol_group]} | {group_dics['complex']['Ion']} & {water_and_solventions_selector[:-1]}\n"
                        elif lig_selector:
                            selector_line += f"{group_dics['complex']['Protein']}|{lig_selector[:-1]}\n"

                        command_local += f'echo -e "{selector_line}q"| {command_name} make_ndx -f {file_name}.gro -o index.ndx\n'


                        with open("tmp.sh", "w") as f:
                            f.write(command_local)
                        subprocess.run("bash tmp.sh", shell=True)
                        os.remove("tmp.sh")

                        group_dics["complex"] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx")

                        if ligand_chains:
                            for ligand_name in ligand_res.values():
                                group_dics[ligand_name] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/{ligand_name}_index.ndx")

                command = f"export GMXLIB=$(pwd)/{md_folder}/FF" + "\n"
                command += f"cd {md_folder}/output_models/{model}/{i}\n"
                local_path = os.getcwd() + f"/{md_folder}/FF"
                command += f'sed -i  "s#{local_path}#$GMXLIB#g" topol/topol.top\n'

                skip_em = os.path.exists(f"{md_folder}/output_models/{model}/{i}/em/prot_em.tpr") and not overwrite
                if not skip_em:
                    command += "mkdir -p em\n"
                    command += "cd em\n"
                    command += f"{remote_command_name} grompp -f ../../../../scripts/em.mdp -c ../topol/{file_name}.gro -p ../topol/topol.top -o prot_em.tpr\n"
                    command += f"{remote_command_name} mdrun -v -deffnm prot_em\n"
                    command += "cd ..\n"

                skip_nvt = os.path.exists(f"{md_folder}/output_models/{model}/{i}/nvt/prot_nvt.tpr") and not overwrite
                if not skip_nvt:
                    command += "mkdir -p nvt\n"
                    command += "cd nvt\n"
                    command += "cp -r ../../../../scripts/nvt.mdp .\n"
                    tc_grps1, tc_grps2 = ["Protein"], "SOL_Ion" if ion_residues else "Water_and_ions"
                    if ion_residues:
                        for r in ion_residues:
                            tc_grps1.append(f"r_{r}")
                            tc_grps2 += f"_&_!r_{r}"
                    if ligand_chains:
                        tc_grps1.extend(ligand_res.values())
                    command += f"sed -i  '/tc-grps/c\\tc-grps = {'_'.join(tc_grps1)} {tc_grps2}' nvt.mdp\n"
                    if ligand_chains:
                        for ligand_name in ligand_res.values():
                            command += f"echo {group_dics[ligand_name]['System_&_!H*']} | {remote_command_name} genrestr -f ../topol/{ligand_name}.acpype/{ligand_name}_GMX.gro -n ../topol/{ligand_name}_index.ndx -o ../topol/{ligand_name}.acpype/posre_{ligand_name}.itp -fc 1000 1000 1000\n"
                    sel = group_dics['complex']["Protein"] if not ion_residues else f"Protein_{'_'.join([f'r_{r}' for r in ion_residues])}"
                    command += f"echo {sel} | {remote_command_name} genrestr -f ../topol/{file_name}.gro -o ../topol/posre.itp -fc 1000 1000 1000 -n ../topol/index.ndx\n"
                    command += f"{remote_command_name} grompp -f nvt.mdp -c ../em/prot_em.gro -p ../topol/topol.top -o prot_nvt.tpr -r ../em/prot_em.gro -n ../topol/index.ndx\n"
                    command += f"{remote_command_name} mdrun -v -deffnm prot_nvt\n"
                    command += "cd ..\n"

                FClist = ("550", "300", "170", "90", "50", "30", "15", "10", "5")
                skip_npt = os.path.exists(f"{md_folder}/output_models/{model}/{i}/npt/prot_npt_{len(FClist)}.tpr") and not overwrite
                if not skip_npt:
                    command += "mkdir -p npt\n"
                    command += "cd npt\n"
                    tc_grps1, tc_grps2 = ["Protein"], "SOL_Ion" if ion_residues else "Water_and_ions"
                    if ion_residues:
                        for r in ion_residues:
                            tc_grps1.append(f"r_{r}")
                            tc_grps2 += f"_&_!r_{r}"
                    if ligand_chains:
                        tc_grps1.extend(ligand_res.values())
                    command += "cp -r ../../../../scripts/npt.mdp .\n"
                    command += f"sed -i  '/tc-grps/c\\tc-grps = {'_'.join(tc_grps1)} {tc_grps2}' npt.mdp\n"
                    sel = group_dics['complex']["Protein"] if not ion_residues else f"Protein_{'_'.join([f'r_{r}' for r in ion_residues])}"
                    for j in range(len(FClist) + 1):
                        if not os.path.exists(f"{md_folder}/output_models/{model}/{i}/npt/prot_npt_{j + 1}.tpr") or overwrite:
                            if j == 0:
                                command += f"{remote_command_name} grompp -f npt.mdp -c ../nvt/prot_nvt.gro -t ../nvt/prot_nvt.cpt -p ../topol/topol.top -o prot_npt_1.tpr -r ../nvt/prot_nvt.gro -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} mdrun -v -deffnm prot_npt_{j + 1}\n"
                            else:
                                if ligand_chains:
                                    for ligand_name in ligand_res.values():
                                        command += f"echo {group_dics[ligand_name]['System_&_!H*']} | {remote_command_name} genrestr -f ../topol/{ligand_name}.acpype/{ligand_name}_GMX.gro -n ../topol/{ligand_name}_index.ndx -o ../topol/{ligand_name}.acpype/posre_{ligand_name}.itp -fc {FClist[j - 1]} {FClist[j - 1]} {FClist[j - 1]}\n"
                                command += f"echo {sel} | {remote_command_name} genrestr -f ../topol/{file_name}.gro -o ../topol/posre.itp -fc {FClist[j - 1]} {FClist[j - 1]} {FClist[j - 1]} -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} grompp -f npt.mdp -c prot_npt_{j}.gro -t prot_npt_{j}.cpt -p ../topol/topol.top -o prot_npt_{j + 1}.tpr -r prot_npt_{j}.gro -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} mdrun -v -deffnm prot_npt_{j + 1}\n"
                    command += "cd ..\n"

                skip_md = os.path.exists(f"{md_folder}/output_models/{model}/{i}/md/prot_md_{frags}.xtc") and not overwrite
                if not skip_md:
                    command += "mkdir -p md\n"
                    command += "cd md\n"
                    tc_grps1, tc_grps2 = ["Protein"], "SOL_Ion" if ion_residues else "Water_and_ions"
                    if ion_residues:
                        for r in ion_residues:
                            tc_grps1.append(f"r_{r}")
                            tc_grps2 += f"_&_!r_{r}"
                    if ligand_chains:
                        tc_grps1.extend(ligand_res.values())
                    command += "cp -r ../../../../scripts/md.mdp .\n"
                    command += f"sed -i  '/tc-grps/c\\tc-grps = {'_'.join(tc_grps1)} {tc_grps2}' md.mdp\n"
                    for j in range(1, frags + 1):
                        if not os.path.exists(f"{md_folder}/output_models/{model}/{i}/md/prot_md_{j}.xtc") or overwrite:
                            if j == 1:
                                command += f"{remote_command_name} grompp -f md.mdp -c ../npt/prot_npt_{len(FClist) + 1}.gro  -t ../npt/prot_npt_{len(FClist) + 1}.cpt -p ../topol/topol.top -o prot_md_{j}.tpr -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} mdrun -v -deffnm prot_md_{j}\n"
                            else:
                                command += f"{remote_command_name} grompp -f md.mdp -c prot_md_{j - 1}.gro -t prot_md_{j - 1}.cpt -p ../topol/topol.top -o prot_md_{j}.tpr -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} mdrun -v -deffnm prot_md_{j}\n"
                    command += "cd ../../../..\n"
                else:
                    command = ''

                if command.strip():
                    jobs.append(command)

        if remove_backups:
            _removeBackupFiles(md_folder)

        return jobs

    def getTrajectoryPaths(self, path, step="md", traj_name="prot_md_cat_noPBC.xtc"):
        """ """
        output_paths = []
        for folder in os.listdir(path + "/output_models/"):
            if folder in self.models_names:
                traj_path = path + "/output_models/" + folder + "/" + step
                output_paths.append(traj_path + "/" + traj_name)

        return output_paths

    def removeBoundaryConditions(self, path, command, step="md", remove_water=False):
        """
        Remove boundary conditions from gromacs simulation trajectory file

        Parameters
        ==========
        path : str
            Path to the job folder where the MD outputs files are located.
        command : str
            Command to call program.
        """
        for folder in os.listdir(path + "/output_models/"):
            if folder in self.models_names:
                traj_path = path + "/output_models/" + folder + "/" + step
                for file in os.listdir(traj_path):
                    if (
                        file.endswith(".xtc")
                        and not file.endswith("_noPBC.xtc")
                        and not os.path.exists(
                            traj_path + "/" + file.split(".")[0] + "_noPBC.xtc"
                        )
                    ):
                        if remove_water == True:
                            option = "14"
                        else:
                            option = "0"
                        os.system(
                            "echo "
                            + option
                            + " | "
                            + command
                            + " trjconv -s "
                            + traj_path
                            + "/"
                            + file.split(".")[0]
                            + ".tpr -f "
                            + traj_path
                            + "/"
                            + file
                            + " -o "
                            + traj_path
                            + "/"
                            + file.split(".")[0]
                            + "_noPBC.xtc -pbc mol -ur compact"
                        )

                if not os.path.exists(traj_path + "/prot_md_cat_noPBC.xtc"):
                    os.system(
                        command
                        + " trjcat -f "
                        + traj_path
                        + "/*_noPBC.xtc -o "
                        + traj_path
                        + "/prot_md_cat_noPBC.xtc -cat"
                    )

                ### md_1 or npt_10

                if (
                    not os.path.exists(
                        "/".join(traj_path.split("/")[:-1])
                        + "/npt/prot_npt_10_no_water.gro"
                    )
                    and remove_water == True
                ):
                    os.system(
                        "echo 1 | gmx editconf -ndef -f "
                        + "/".join(traj_path.split("/")[:-1])
                        + "/npt/prot_npt_10.gro -o "
                        + "/".join(traj_path.split("/")[:-1])
                        + "/npt/prot_npt_10_no_water.gro"
                    )

    def setUpOpenMMSimulations(self, job_folder, replicas, simulation_time, ligand_charges=None, residue_names=None, ff='amber14',
                               add_bonds=None, skip_ligands=None, ligand_only=None, metal_ligand=None, metal_parameters=None, skip_replicas=None,
                               extra_frcmod=None, extra_mol2=None, dcd_report_time=100.0, data_report_time=100.0,
                               non_standard_residues=None, add_hydrogens=True, extra_force_field=None,
                               nvt_time=0.1, npt_time=0.2, nvt_temp_scaling_steps=50, npt_restraint_scaling_steps=50,
                               restraint_constant=5.0, chunk_size=100.0,
                               equilibration_data_report_time=1.0, equilibration_dcd_report_time=0.0, temperature=300.0,
                               collision_rate=1.0, time_step=0.002, cuda=False, fixed_seed=None, script_file=None,
                               extra_script_options=None, add_counterionsRand=False, skip_preparation=False,
                               solvate=True,
                               verbose=False,
                               strict_ligand_atom_check=True,
                               ligand_parameters_source=None,
                               parameterization_method='ambertools',
                               parameterization_options=None,
                               only_models=None, skip_models=None,
                               ligand_smiles=None,
                               ligand_sdf_files=None,
                               ligand_xml_files=None,
                               skip_ligand_charge_computation=False,
                               export_per_residue_ffxml=False,
                               ):
        """
        Set up OpenMM simulations for multiple models with customizable ligand charges, residue names, and force field options.
        Includes support for multiple replicas.

        Parameters:
        ...
        - nvt_temp_scaling_steps (int, optional): Temperature scaling segments during NVT equilibration.
          Must be at least 1 and cannot exceed the number of MD steps derived from `nvt_time` and `time_step`.
        - npt_restraint_scaling_steps (int, optional): Restraint scaling segments during NPT equilibration.
          Must be at least 1 and cannot exceed the number of MD steps derived from `npt_time` and `time_step`.
        - skip_preparation (bool, optional): If True, skip preparation steps (e.g., ligand parameterization and input generation)
          but still return simulation jobs for the models/replicas. This allows running with pre-existing inputs.
        - solvate (bool, optional): When False, request the backends to skip solvent/ion addition (defaults to True).
        - verbose (bool, optional): When True, emit detailed progress information from the parameterization backends.
        - strict_ligand_atom_check (bool, optional): If True (default), ensure parameter packs or extra MOL2 inputs use the
          same atom-name set as the ligand extracted from the PDB; mismatches raise an error. Disable to downgrade to warnings.
        - ligand_only (bool or str or list of str, optional): If set, skip protein + ligand simulations and instead
          prepare solvated ligand-only systems. Use True to process every ligand detected in each model, a single
          residue name to limit the run to that ligand, or a list of residue names for multiple ligands.
        - ligand_parameters_source (str, optional): Path to a folder containing per-ligand parameter packs named
          like `<RESNAME>_parameters` (e.g., `NAD_parameters`). You can pass either the folder that directly
          contains these packs, or a parent folder with a `parameters/` subfolder. Only ligands present in the
          system are considered; for those found, the full pack is copied into the job `parameters/` and
          parameterization for that ligand is skipped via `metal_parameters`.
        - parameterization_method (str, optional): Name of the parameterization backend to use. Defaults to
          ``'ambertools'`` which preserves the existing workflow.
        - parameterization_options (dict, optional): Backend-specific options forwarded to the selected
          parameterization backend.
        - ligand_smiles (dict, optional): Mapping from ligand residue name to SMILES strings. Required by the
          OpenFF backend to build ligand templates; ignored by AmberTools.
        - ligand_sdf_files (dict, optional): Mapping from residue name to SDF file paths or option dictionaries
          (with optional ``index`` and ``charge_method`` keys). When provided, the OpenFF backend uses the SDF
          connectivity/chemistry while retaining ligand coordinates from the model PDB.
        - ligand_xml_files (dict, optional): Mapping from residue name to FFXML files containing templates to add
          to the OpenFF forcefield (e.g., {'HEM': 'path/HEM.ffxml'}). Ignored by the AmberTools backend.
        - only_models (str or list of str, optional): If provided, restrict setup to these model names only.
        - skip_models (list of str, optional): If provided, skip setup for these models.
        - skip_ligand_charge_computation (bool, optional): If True, the parameterization backends reuse
          existing ligand charges (OpenFF expects cached ``<RES>.offmol.json`` templates, AmberTools relies on
          existing MOL2/PREPI files). Missing caches trigger debugging placeholders for OpenFF (zero charges with
          a warning) and hard errors for AmberTools to guard realistic runs.
        - export_per_residue_ffxml (bool or str or list of str, optional): If True, export FFXML templates for all
          ligands prepared by the OpenFF backend into the parameters folder; if a string or list, export only for the
          specified residue names. The AmberTools backend will also export when requested, using available mol2/frcmod
          inputs for the selected residues.
        """

        openmm_setup = _require_openmm_support("proteinModels.setUpOpenMMSimulations")
        openmm_md_cls = getattr(openmm_setup, "openmm_md", None)
        if openmm_md_cls is None:
            raise ImportError(
                "OpenMM support is enabled but the 'openmm_md' class could not be located. "
                "Reinstall prepare_proteins with its OpenMM extras."
            )

        import os, shutil

        if ligand_smiles is not None:
            if not isinstance(ligand_smiles, dict):
                raise TypeError("ligand_smiles must be a dict mapping residue name to SMILES.")
            normalized_smiles = {}
            for key, value in ligand_smiles.items():
                if not isinstance(key, str):
                    raise TypeError("ligand_smiles keys must be strings.")
                if not isinstance(value, str):
                    raise TypeError(f"SMILES for residue {key!r} must be a string.")
                residue_name = key.strip().upper()
                smiles_value = value.strip()
                if not residue_name or not smiles_value:
                    raise ValueError("ligand_smiles entries must not be empty.")
                normalized_smiles[residue_name] = smiles_value
            ligand_smiles = normalized_smiles

        if ligand_sdf_files is not None:
            if not isinstance(ligand_sdf_files, dict):
                raise TypeError("ligand_sdf_files must be a dict mapping residue name to SDF definitions.")
            normalized_sdfs = {}
            for key, value in ligand_sdf_files.items():
                if not isinstance(key, str):
                    raise TypeError("ligand_sdf_files keys must be strings.")
                residue_name = key.strip().upper()
                if not residue_name:
                    raise ValueError("ligand_sdf_files entries must not have empty residue names.")
                if isinstance(value, str):
                    normalized_sdfs[residue_name] = {"path": value}
                elif isinstance(value, dict):
                    if "path" not in value:
                        raise ValueError(f"ligand_sdf_files entry for {residue_name!r} is missing a 'path'.")
                    entry = {"path": value["path"]}
                    if "index" in value:
                        entry["index"] = value["index"]
                    if "charge_method" in value:
                        entry["charge_method"] = value["charge_method"]
                    normalized_sdfs[residue_name] = entry
                else:
                    raise TypeError(
                        "ligand_sdf_files values must be path strings or dictionaries with options."
                    )
            ligand_sdf_files = normalized_sdfs
        if ligand_xml_files is not None:
            if not isinstance(ligand_xml_files, dict):
                raise TypeError("ligand_xml_files must be a dict mapping residue name to FFXML paths.")
            normalized_xml = {}
            for key, value in ligand_xml_files.items():
                if not isinstance(key, str):
                    raise TypeError("ligand_xml_files keys must be residue name strings.")
                resname = key.strip().upper()
                if not resname:
                    continue
                if not isinstance(value, (str, os.PathLike)):
                    raise TypeError(f"ligand_xml_files[{key!r}] must be a path string.")
                path_str = os.fspath(value)
                normalized_xml[resname] = path_str
            ligand_xml_files = normalized_xml

        # Normalize export flag (OpenFF-only)
        if isinstance(export_per_residue_ffxml, str):
            export_per_residue_ffxml = [export_per_residue_ffxml]
        elif isinstance(export_per_residue_ffxml, (list, tuple, set)):
            export_per_residue_ffxml = [str(v) for v in export_per_residue_ffxml]

        def _select_backend_parameters_folder(base_folder: str, backend_label: str) -> str:
            if not backend_label:
                return base_folder
            backend_folder = os.path.join(base_folder, backend_label)
            if backend_label == "ambertools":
                try:
                    entries = os.listdir(base_folder)
                except OSError:
                    entries = []
                has_legacy_packs = any(
                    entry.endswith('_parameters') and os.path.isdir(os.path.join(base_folder, entry))
                    for entry in entries
                )
                if has_legacy_packs:
                    return base_folder
            os.makedirs(backend_folder, exist_ok=True)
            return backend_folder

        # Normalize model filters
        if isinstance(only_models, str):
            only_models = [only_models]
        if skip_models is None:
            skip_models = []

        def _replica_has_inputs(replica_folder: str) -> bool:
            input_dir = os.path.join(replica_folder, 'input_files')
            if not os.path.isdir(input_dir):
                return False
            files = os.listdir(input_dir)
            has_prmtop = any(f.endswith('.prmtop') for f in files)
            has_inpcrd = any(f.endswith('.inpcrd') or f.endswith('.rst7') for f in files)
            return has_prmtop and has_inpcrd

        def _replicas_needed(model_folder: str, replicas: int, skip_replicas):
            zfill = max(len(str(replicas)), 2)
            needed, prepared = [], []
            for replica in range(1, replicas + 1):
                if skip_replicas and replica in skip_replicas:
                    continue
                rname = f"replica_{str(replica).zfill(zfill)}"
                rfolder = os.path.join(model_folder, rname)
                if _replica_has_inputs(rfolder):
                    prepared.append(replica)
                else:
                    needed.append(replica)
            return needed, prepared

        def setUpJobs(job_folder, openmm_md, script_file, parameterization_result=None, simulation_time=simulation_time,
                      dcd_report_time=dcd_report_time, data_report_time=data_report_time,
                      nvt_time=nvt_time, npt_time=npt_time, nvt_temp_scaling_steps=nvt_temp_scaling_steps,
                      npt_restraint_scaling_steps=npt_restraint_scaling_steps,
                      restraint_constant=restraint_constant, chunk_size=chunk_size,
                      equilibration_data_report_time=equilibration_data_report_time,
                      equilibration_dcd_report_time=equilibration_dcd_report_time, temperature=temperature,
                      collision_rate=collision_rate, time_step=time_step, cuda=cuda,
                      fixed_seed=fixed_seed, add_counterionsRand=add_counterionsRand):
            """Set up simulation jobs for a single replica.

            This prepares the replica folder with an `input_files/` directory containing
            the required simulation inputs. For the current AMBER workflow, this means
            copying the `.prmtop` and `.inpcrd`/`.rst7` files and returning a list with
            the command to run the OpenMM simulation script.
            If a ``parameterization_result`` is provided, its metadata determines which
            files are copied into the replica folder.

            Returns
            -------
            list[str]
                Shell command(s) to run for this replica.
            """

            jobs: list[str] = []

            # Ensure replica folder and inputs folder exist
            os.makedirs(job_folder, exist_ok=True)
            input_dir = os.path.join(job_folder, 'input_files')
            os.makedirs(input_dir, exist_ok=True)

            # Base filename for input files (model basename + extension)
            base_name = getattr(openmm_md, 'pdb_name', 'input')
            resolved_parameterization = parameterization_result or getattr(openmm_md, 'parameterization_result', None)
            input_format = None
            if resolved_parameterization is not None:
                input_format = resolved_parameterization.input_format
                prmtop_src = resolved_parameterization.prmtop_path or getattr(openmm_md, 'prmtop_file', None)
                inpcrd_src = resolved_parameterization.coordinates_path or getattr(openmm_md, 'inpcrd_file', None)
                openmm_md.parameterization_result = resolved_parameterization
            else:
                prmtop_src = getattr(openmm_md, 'prmtop_file', None)
                inpcrd_src = getattr(openmm_md, 'inpcrd_file', None)

            if input_format and input_format != 'amber':
                raise NotImplementedError(
                    f"Parameterization format '{input_format}' is not supported by the default OpenMM runner."
                )

            prmtop_ext = os.path.splitext(prmtop_src)[1] if prmtop_src else '.prmtop'
            inpcrd_ext = os.path.splitext(inpcrd_src)[1] if inpcrd_src else '.inpcrd'

            prmtop_name = f"{base_name}{prmtop_ext}"
            inpcrd_name = f"{base_name}{inpcrd_ext}"

            prmtop_path = os.path.join(input_dir, prmtop_name)
            inpcrd_path = os.path.join(input_dir, inpcrd_name)

            # Copy source files into input directory using the model's basename
            if prmtop_src and os.path.exists(prmtop_src) and not os.path.exists(prmtop_path):
                shutil.copyfile(prmtop_src, prmtop_path)
            if inpcrd_src and os.path.exists(inpcrd_src) and not os.path.exists(inpcrd_path):
                shutil.copyfile(inpcrd_src, inpcrd_path)

            # Build the command regardless; execution-time checks will fail fast if inputs are missing
            cmd = []
            cmd.append(f"cd {job_folder}")

            # Use a path to the script relative to the replica folder so the
            # simulation script can be located regardless of the submission
            # working directory.
            rel_script = os.path.relpath(script_file, job_folder)

            cmd.append(
                "python "
                + f"{rel_script} "
                + f"{os.path.join('input_files', prmtop_name)} "
                + f"{os.path.join('input_files', inpcrd_name)} "
                + f"{simulation_time} "
                + f"--chunk_size {chunk_size} "
                + f"--temperature {temperature} "
                + f"--collision_rate {collision_rate} "
                + f"--time_step {time_step} "
                + f"--dcd_report_time {dcd_report_time} "
                + f"--data_report_time {data_report_time} "
                + f"--nvt_time {nvt_time} "
                + f"--npt_time {npt_time} "
                + f"--nvt_temp_scaling_steps {nvt_temp_scaling_steps} "
                + f"--npt_restraint_scaling_steps {npt_restraint_scaling_steps} "
                + f"--restraint_constant {restraint_constant} "
                + f"--equilibration_data_report_time {equilibration_data_report_time} "
                + f"--equilibration_dcd_report_time {equilibration_dcd_report_time}"
            )
            # Always provide a seed for the simulation. Use the given fixed seed
            # when supplied; otherwise generate a random one so the OpenMM
            # script, which requires the flag, always receives a value.
            seed = int(fixed_seed) if fixed_seed is not None else int(np.random.randint(0, 2**31 - 1))
            cmd[-1] += f" --seed {seed}"
            jobs.append("\n".join(cmd) + "\n")

            return jobs

        # Create the base job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        self.openmm_md = _OpenMMSimulationRegistry()

        ligand_parameters_folder = os.path.join(job_folder, 'parameters')
        os.makedirs(ligand_parameters_folder, exist_ok=True)

        if ligand_parameters_source is not None and not os.path.exists(ligand_parameters_source):
            raise ValueError(f"ligand_parameters_source not found: {ligand_parameters_source}")

        # Resolve the actual source directory that contains `<RES>_parameters` packs.
        def _resolve_ligand_pack_root(path, backend_label=None):
            if path is None:
                return None
            candidate_paths = []
            if backend_label:
                candidate_paths.append(os.path.join(path, backend_label))
            candidate_paths.append(path)
            for candidate in candidate_paths:
                try:
                    entries = os.listdir(candidate)
                except Exception:
                    entries = []
                has_packs = any(
                    name.endswith('_parameters') and os.path.isdir(os.path.join(candidate, name)) for name in entries
                )
                if has_packs:
                    return candidate
                sub = os.path.join(candidate, 'parameters')
                if not os.path.isdir(sub):
                    continue
                try:
                    sub_entries = os.listdir(sub)
                except Exception:
                    continue
                has_packs = any(
                    name.endswith('_parameters') and os.path.isdir(os.path.join(sub, name)) for name in sub_entries
                )
                if has_packs:
                    return sub
            return None
        resolved_ligand_pack_root = None
        backend_options = dict(parameterization_options) if parameterization_options else {}
        if ligand_smiles:
            backend_options.setdefault("ligand_smiles", ligand_smiles)
        if ligand_sdf_files:
            backend_options.setdefault("ligand_sdf_files", ligand_sdf_files)
        if skip_ligand_charge_computation:
            backend_options.setdefault("skip_ligand_charge_computation", True)
        backend_options.setdefault("solvate", bool(solvate))
        if "verbose" not in backend_options:
            backend_options["verbose"] = bool(verbose)
        backend_name_lower = str(parameterization_method).lower()
        if backend_name_lower == "openff":
            if ligand_xml_files:
                backend_options.setdefault("ligand_xml_files", ligand_xml_files)
            if export_per_residue_ffxml:
                backend_options.setdefault("export_per_residue_ffxml", export_per_residue_ffxml)
        elif backend_name_lower == "ambertools":
            if ligand_xml_files:
                backend_options.setdefault("ligand_xml_files", ligand_xml_files)
        backend = get_backend(parameterization_method, **backend_options)
        backend_label = getattr(backend, "name", str(parameterization_method)).lower()
        ligand_parameters_folder = _select_backend_parameters_folder(ligand_parameters_folder, backend_label)
        resolved_ligand_pack_root = _resolve_ligand_pack_root(ligand_parameters_source, backend_label)

        script_folder = os.path.join(job_folder, 'scripts')
        if not os.path.exists(script_folder):
            os.mkdir(script_folder)

        if not script_file:
            _copyScriptFile(script_folder, "openmm_simulation.py", subfolder='md/openmm', hidden=False)
            script_file = script_folder + '/openmm_simulation.py'

        aa3_residues = set(openmm_setup.aa3)
        ALL_LIGANDS = "__ALL__"

        def _normalize_ligand_only_option(option):
            if option in (None, False):
                return None
            if option is True:
                return ALL_LIGANDS
            if isinstance(option, str):
                value = option.strip()
                if not value:
                    return None
                if value.lower() == "all":
                    return ALL_LIGANDS
                return {value.upper()}
            if isinstance(option, (list, tuple, set)):
                normalized = {str(item).strip().upper() for item in option if str(item).strip()}
                if not normalized:
                    return None
                if "ALL" in normalized:
                    return ALL_LIGANDS
                return normalized
            raise TypeError("ligand_only must be True, False, a string, or an iterable of strings.")

        ligand_only_selector = _normalize_ligand_only_option(ligand_only)
        ligand_only_active = ligand_only_selector is not None

        skip_ligands_upper = set(DEFAULT_PARAMETERIZATION_SKIP_RESIDUES)
        if skip_ligands:
            if isinstance(skip_ligands, str):
                skip_ligands_upper.add(skip_ligands.strip().upper())
            else:
                skip_ligands_upper.update({str(item).strip().upper() for item in skip_ligands})
        effective_skip_ligands = sorted(skip_ligands_upper) if skip_ligands_upper else None

        def _detect_model_ligands(model_name):
            structure = self.structures.get(model_name)
            if structure is None:
                return []
            ligands = []
            for residue in structure.get_residues():
                resname = getattr(residue, 'resname', '').strip().upper()
                if not resname or resname in ('HOH', 'WAT'):
                    continue
                if resname in aa3_residues or resname in skip_ligands_upper:
                    continue
                if resname not in ligands:
                    ligands.append(resname)
            return ligands

        def _collect_ligand_packs(parameters_folder):
            packs = {}
            if not os.path.isdir(parameters_folder):
                return packs
            for entry in os.listdir(parameters_folder):
                if entry.endswith('_parameters'):
                    residue = entry[: -len('_parameters')]
                    packs[residue.upper()] = os.path.join(parameters_folder, entry)
            return packs

        def _posix_path(path):
            return Path(path).expanduser().absolute().as_posix()

        def _execute_with_logging(command, model_key=None):
            entry = None
            if model_key is not None:
                md_obj = self.openmm_md.get(model_key)
                if md_obj is not None:
                    log = getattr(md_obj, "command_log", None)
                    if log is not None:
                        entry = {"command": command.rstrip(), "returncode": None}
                        log.append(entry)
            ret = os.system(command)
            if entry is not None:
                entry["returncode"] = ret
            return ret

        def _ensure_ligand_only_inputs(model_name, ligand_name, packs, allow_generate):
            ligand_key = ligand_name.upper()
            base_tag = f"{model_name}_{ligand_key}"
            prmtop_path = os.path.join(ligand_parameters_folder, f"{base_tag}.prmtop")
            inpcrd_path = os.path.join(ligand_parameters_folder, f"{base_tag}.inpcrd")
            inpcrd_candidates = [inpcrd_path, os.path.join(ligand_parameters_folder, f"{base_tag}.rst7")]
            existing_inpcrd = next((candidate for candidate in inpcrd_candidates if os.path.exists(candidate)), None)
            model_ligand_pdb = os.path.join(ligand_parameters_folder, f"{base_tag}.pdb")
            if os.path.exists(prmtop_path) and existing_inpcrd:
                return prmtop_path, existing_inpcrd

            if not allow_generate:
                raise FileNotFoundError(
                    f"Missing ligand-only inputs for {ligand_key} in model {model_name}. "
                    f"Enable preparation or place the AMBER files in {ligand_parameters_folder}."
                )

            pack_dir = packs.get(ligand_key)
            if not pack_dir or not os.path.isdir(pack_dir):
                packs.update(_collect_ligand_packs(ligand_parameters_folder))
                pack_dir = packs.get(ligand_key)
            if not pack_dir or not os.path.isdir(pack_dir):
                raise ValueError(
                    f"Parameter pack for ligand {ligand_key} not found in {ligand_parameters_folder}. "
                    "Run preparation without skip_preparation or provide ligand_parameters_source."
                )

            ligand_pdb = os.path.join(pack_dir, f"{ligand_key}.pdb")
            if os.path.exists(model_ligand_pdb):
                ligand_pdb = model_ligand_pdb
            else:
                if not os.path.exists(ligand_pdb):
                    raise FileNotFoundError(f"Ligand PDB for {ligand_key} not found at {ligand_pdb}.")
                shutil.copyfile(ligand_pdb, model_ligand_pdb)
                ligand_pdb = model_ligand_pdb

            mol2_path = os.path.join(ligand_parameters_folder, f"{ligand_key}.mol2")
            mol2_manifest = os.path.join(ligand_parameters_folder, f"{ligand_key}.mol2.list")
            alt_mol2 = os.path.join(pack_dir, f"{ligand_key}.mol2")
            alt_mol2_manifest = os.path.join(pack_dir, f"{ligand_key}.mol2.list")
            if os.path.exists(alt_mol2_manifest):
                shutil.copyfile(alt_mol2_manifest, mol2_manifest)
            if os.path.exists(alt_mol2):
                shutil.copyfile(alt_mol2, mol2_path)
            elif not os.path.exists(mol2_path):
                raise FileNotFoundError(f"Mol2 file for ligand {ligand_key} not found in {ligand_parameters_folder} or pack {pack_dir}.")
            mol2_paths = []
            if os.path.exists(mol2_manifest):
                try:
                    with open(mol2_manifest) as mf:
                        for raw in mf:
                            entry = raw.strip()
                            if not entry:
                                continue
                            if os.path.isabs(entry):
                                candidate = entry
                            else:
                                candidate = os.path.normpath(os.path.join(ligand_parameters_folder, entry))
                            if os.path.exists(candidate):
                                mol2_paths.append(candidate)
                except OSError:
                    mol2_paths = []
            if not mol2_paths:
                mol2_paths = [mol2_path]

            frcmod_path = os.path.join(ligand_parameters_folder, f"{ligand_key}.frcmod")
            alt_frcmod = os.path.join(pack_dir, f"{ligand_key}.frcmod")
            if os.path.exists(alt_frcmod):
                shutil.copyfile(alt_frcmod, frcmod_path)
            elif not os.path.exists(frcmod_path):
                raise FileNotFoundError(f"Frcmod file for ligand {ligand_key} not found in {ligand_parameters_folder} or pack {pack_dir}.")
            frcmod_manifest = os.path.join(ligand_parameters_folder, f"{ligand_key}.frcmod.list")
            alt_manifest = os.path.join(pack_dir, f"{ligand_key}.frcmod.list")
            if os.path.exists(alt_manifest):
                shutil.copyfile(alt_manifest, frcmod_manifest)
            frcmod_paths = []
            if os.path.exists(frcmod_manifest):
                try:
                    with open(frcmod_manifest) as mf:
                        for raw in mf:
                            entry = raw.strip()
                            if not entry:
                                continue
                            if os.path.isabs(entry):
                                candidate = entry
                            else:
                                candidate = os.path.normpath(os.path.join(ligand_parameters_folder, entry))
                            if os.path.exists(candidate):
                                frcmod_paths.append(candidate)
                except OSError:
                    frcmod_paths = []
            if not frcmod_paths:
                frcmod_paths = [frcmod_path]

            tleap_script = os.path.join(ligand_parameters_folder, f"{base_tag}_ligand_only.leap")
            with open(tleap_script, 'w') as tlf:
                tlf.write('source leaprc.gaff\n')
                tlf.write('source leaprc.water.tip3p\n')
                for frcmod in frcmod_paths:
                    tlf.write(f'loadamberparams "{_posix_path(frcmod)}"\n')
                primary_mol2 = mol2_paths[0]
                tlf.write(f'{ligand_key} = loadmol2 "{_posix_path(primary_mol2)}"\n')
                tlf.write(f'mol = loadpdb "{_posix_path(ligand_pdb)}"\n')
                tlf.write('solvatebox mol TIP3PBOX 12\n')
                if add_counterionsRand:
                    tlf.write('addIonsRand mol Na+ 0\n')
                    tlf.write('addIonsRand mol Cl- 0\n')
                else:
                    tlf.write('addIons2 mol Na+ 0\n')
                    tlf.write('addIons2 mol Cl- 0\n')
                tlf.write(f'saveamberparm mol "{_posix_path(prmtop_path)}" "{_posix_path(inpcrd_path)}"\n')
                tlf.write('quit\n')

            command = f'tleap -s -f "{tleap_script}"'
            ret = _execute_with_logging(command, model_name)
            if ret != 0:
                raise RuntimeError(f"tleap failed while preparing ligand-only inputs for {ligand_key} in model {model_name}.")

            existing_inpcrd = next((candidate for candidate in inpcrd_candidates if os.path.exists(candidate)), None)
            if not (os.path.exists(prmtop_path) and existing_inpcrd):
                raise RuntimeError(f"Failed to generate prmtop/inpcrd for ligand {ligand_key} in model {model_name}.")

            return prmtop_path, existing_inpcrd

        def _prepare_ligand_only_jobs(model_name, selected_ligands, packs):
            jobs = []
            allow_generate = not skip_preparation
            zfill = max(len(str(replicas)), 2)

            for ligand_name in selected_ligands:
                prmtop_src, inpcrd_src = _ensure_ligand_only_inputs(model_name, ligand_name, packs, allow_generate)

                ligand_root = os.path.join(job_folder, ligand_name)
                os.makedirs(ligand_root, exist_ok=True)

                model_root = os.path.join(ligand_root, model_name)
                os.makedirs(model_root, exist_ok=True)

                ligand_md = SimpleNamespace(
                    pdb_name=f"{model_name}_{ligand_name.upper()}",
                    prmtop_file=prmtop_src,
                    inpcrd_file=inpcrd_src,
                )

                for replica in range(1, replicas + 1):
                    if skip_replicas and replica in skip_replicas:
                        continue
                    replica_folder = os.path.join(model_root, f"replica_{str(replica).zfill(zfill)}")
                    jobs.extend(setUpJobs(replica_folder, ligand_md, script_file))

            return jobs

        simulation_jobs = []
        ligand_simulation_jobs = []
        ligand_setup_completed = False

        for model in self:
            if only_models and model not in only_models:
                continue
            if model in skip_models:
                continue
            if ligand_only_active and ligand_setup_completed:
                break

            model_ligands = _detect_model_ligands(model)

            self.openmm_md[model] = openmm_md_cls(self.models_paths[model])
            if not skip_preparation:
                self.openmm_md[model].setUpFF(ff)
                if add_hydrogens:
                    variants = self.openmm_md[model].getProtonationStates()
                    self.openmm_md[model].removeHydrogens()
                    self.openmm_md[model].addHydrogens(variants=variants)

            if ligand_only_active:
                if ligand_only_selector == ALL_LIGANDS:
                    selected_ligands = model_ligands
                else:
                    selected_ligands = [lig for lig in model_ligands if lig in ligand_only_selector]
                    missing = ligand_only_selector - set(selected_ligands)
                    if missing:
                        raise ValueError(
                            f"Requested ligand(s) {sorted(missing)} not found in model {model}. "
                            f"Available ligands: {model_ligands or 'none'}"
                        )

                if not selected_ligands:
                    continue

                selected_ligands = [lig.upper() for lig in selected_ligands]
                selected_ligands_set = set(selected_ligands)

                packs = _collect_ligand_packs(ligand_parameters_folder)
                missing_packs = [lig for lig in selected_ligands if lig not in packs]
                if missing_packs and skip_preparation:
                    raise FileNotFoundError(
                        f"Ligand parameter packs missing for {missing_packs} in model {model}. "
                        "Disable skip_preparation or provide pre-generated packs."
                    )

                if missing_packs and not skip_preparation:
                    external_params = metal_parameters if metal_parameters else resolved_ligand_pack_root
                    skip_for_param = set(skip_ligands_upper)
                    if ligand_only_selector != ALL_LIGANDS:
                        skip_for_param.update(
                            lig.upper() for lig in model_ligands if lig.upper() not in selected_ligands_set
                        )
                    skip_argument = list(skip_for_param) if skip_for_param else None
                prepare_kwargs = dict(
                    charges=ligand_charges,
                    metal_ligand=metal_ligand,
                    add_bonds=add_bonds.get(model) if add_bonds else None,
                    skip_ligands=sorted(skip_argument) if skip_argument else effective_skip_ligands,
                    overwrite=False,
                    metal_parameters=external_params,
                    extra_frcmod=extra_frcmod,
                    extra_mol2=extra_mol2,
                    cpus=20,
                    return_qm_jobs=True,
                    extra_force_field=extra_force_field,
                    force_field='ff14SB',
                    residue_names=residue_names.get(model) if residue_names else None,
                    add_counterions=True,
                    add_counterionsRand=add_counterionsRand,
                    save_amber_pdb=True,
                    solvate=bool(solvate),
                    regenerate_amber_files=True,
                    non_standard_residues=non_standard_residues,
                    strict_atom_name_check=strict_ligand_atom_check,
                    only_residues=selected_ligands_set,
                    build_full_system=False,
                    ligand_sdf_files=ligand_sdf_files,
                )
                if getattr(backend, "name", "").lower() == "openff":
                    prepare_kwargs["ligand_xml_files"] = ligand_xml_files
                elif getattr(backend, "name", "").lower() == "ambertools":
                    prepare_kwargs["ligand_xml_files"] = ligand_xml_files
                prepare_kwargs["export_per_residue_ffxml"] = export_per_residue_ffxml
                if "verbose" not in prepare_kwargs:
                    prepare_kwargs["verbose"] = bool(verbose)

                backend.prepare_model(
                    self.openmm_md[model],
                    ligand_parameters_folder,
                    **prepare_kwargs,
                )
                self.openmm_md[model].parameterization_result = backend.describe_model(self.openmm_md[model])
                packs = _collect_ligand_packs(ligand_parameters_folder)

                ligand_simulation_jobs.extend(_prepare_ligand_only_jobs(model, selected_ligands, packs))
                ligand_setup_completed = True
                continue

            model_folder = os.path.join(job_folder, model)
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)

            needed_replicas, prepared_replicas = _replicas_needed(model_folder, replicas, skip_replicas)

            base_name = getattr(self.openmm_md[model], 'pdb_name', 'input')
            job_prmtop = os.path.join(ligand_parameters_folder, f"{base_name}.prmtop")
            job_inpcrd = None
            for ext in ('.inpcrd', '.rst7'):
                candidate = os.path.join(ligand_parameters_folder, f"{base_name}{ext}")
                if os.path.exists(candidate):
                    job_inpcrd = candidate
                    break
            if os.path.exists(job_prmtop) and job_inpcrd:
                self.openmm_md[model].prmtop_file = job_prmtop
                self.openmm_md[model].inpcrd_file = job_inpcrd

            if not getattr(self.openmm_md[model], 'prmtop_file', None) or not getattr(self.openmm_md[model], 'inpcrd_file', None):
                if prepared_replicas:
                    zfill_pre = max(len(str(replicas)), 2)
                    first_prepared = prepared_replicas[0]
                    rep_name = f"replica_{str(first_prepared).zfill(zfill_pre)}"
                    rep_input = os.path.join(model_folder, rep_name, 'input_files')
                    if os.path.isdir(rep_input):
                        prmtops = [f for f in os.listdir(rep_input) if f.endswith('.prmtop')]
                        inpcrds = [f for f in os.listdir(rep_input) if f.endswith('.inpcrd') or f.endswith('.rst7')]
                        if prmtops and inpcrds:
                            self.openmm_md[model].prmtop_file = os.path.join(rep_input, prmtops[0])
                            self.openmm_md[model].inpcrd_file = os.path.join(rep_input, inpcrds[0])

            has_prmtop = bool(getattr(self.openmm_md[model], 'prmtop_file', None))
            has_inpcrd = bool(getattr(self.openmm_md[model], 'inpcrd_file', None))
            if (not skip_preparation) and (len(needed_replicas) > 0) and not (has_prmtop and has_inpcrd):
                external_params = metal_parameters if metal_parameters else resolved_ligand_pack_root
                prepare_kwargs = dict(
                    charges=ligand_charges,
                    metal_ligand=metal_ligand,
                    add_bonds=add_bonds.get(model) if add_bonds else None,
                    skip_ligands=effective_skip_ligands,
                    overwrite=False,
                    metal_parameters=external_params,
                    extra_frcmod=extra_frcmod,
                    extra_mol2=extra_mol2,
                    cpus=20,
                    return_qm_jobs=True,
                    extra_force_field=extra_force_field,
                    force_field='ff14SB',
                    residue_names=residue_names.get(model) if residue_names else None,
                    add_counterions=True,
                    add_counterionsRand=add_counterionsRand,
                    save_amber_pdb=True,
                    solvate=bool(solvate),
                    regenerate_amber_files=True,
                    non_standard_residues=non_standard_residues,
                    strict_atom_name_check=strict_ligand_atom_check,
                    ligand_sdf_files=ligand_sdf_files,
                )
                if getattr(backend, "name", "").lower() == "openff":
                    prepare_kwargs["ligand_xml_files"] = ligand_xml_files
                elif getattr(backend, "name", "").lower() == "ambertools":
                    prepare_kwargs["ligand_xml_files"] = ligand_xml_files
                prepare_kwargs["export_per_residue_ffxml"] = export_per_residue_ffxml
                if "verbose" not in prepare_kwargs:
                    prepare_kwargs["verbose"] = bool(verbose)

                backend.prepare_model(
                    self.openmm_md[model],
                    ligand_parameters_folder,
                    **prepare_kwargs,
                )

            parameterization_result = backend.describe_model(self.openmm_md[model])
            self.openmm_md[model].parameterization_result = parameterization_result

            zfill = max(len(str(replicas)), 2)
            for replica in range(1, replicas + 1):
                if skip_replicas and replica in skip_replicas:
                    continue

                replica_str = str(replica).zfill(zfill)
                replica_folder = os.path.join(model_folder, f'replica_{replica_str}')
                if not os.path.exists(replica_folder):
                    os.mkdir(replica_folder)

                jobs = setUpJobs(
                    replica_folder,
                    self.openmm_md[model],
                    script_file,
                    parameterization_result=parameterization_result,
                )
                simulation_jobs.extend(jobs)

        self.openmm_md.openmm_command_logs.clear()
        for model_name, md_obj in self.openmm_md.items():
            self.openmm_md.openmm_command_logs[model_name] = list(getattr(md_obj, "command_log", []))

        if ligand_only_active:
            return ligand_simulation_jobs

        return simulation_jobs

    def setUpPLACERcalculation(self, PLACERfolder, output_folder="output_folder", PLACER_PATH="/gpfs/projects/bsc72/conda_envs/PLACER/", suffix=None, num_samples=50,
                           ligand=None, apo=False,rerank="prmsd", mutate=None, mutate_chain="A",
                           mutate_to=None, residue_json=None):
        """
        Set up PLACER calculations for evaluating catalytic centers, with or without ligand.
        Special amino acids can be added.

        Visit https://github.com/baker-laboratory/PLACER/tree/main for more options.

        Parameters
        ----------
        PLACERfolder : str
            Directory where all job-related files will be stored.
        output_folder : str, default="output_folder"
            Folder containing PDB files to run.
        suffix : str, optional
            Suffix added to output PDB file.
        num_samples : int, default=50
            Number of samples to generate. 50-100 is a good number in most cases.
        ligand : str, optional
            Ligand <name3>, <name3-resno>, or <chain-name3-resno> (e.g., "L-LIG-1") to be predicted.
            All other ligands will be fixed.
            If not specified, PLACER will detect the ligand automatically.
        apo : bool, default=False
            run PLACER in apo mode:
        rerank : str, optional
            Rank models using one of the input metrics: "prmsd", "plddt", or "plddt_pde".
            "prmsd" is sorted in ascending order, "plddt" and "plddt_pde" in descending order.
        mutate : dict, optional
            Dictionary with model names as keys and residue numbers to mutate as values.
            Requires `mutate_to` to be specified.
        mutate_chain : str, default="A"
            Chain where the residues to mutate are located.
        mutate_to : str, optional
            Residue name (3-letter code) to mutate into.
        residue_json : str, optional
            JSON file specifying custom residues used in the PDB or with `mutate`.

        Returns
        -------
        list
            List of command-line commands to be executed.
        """
        # validate PLACER_PATH option
        valid_PLACER_PATH = {"/gpfs/projects/bsc72/conda_envs/PLACER/","/gpfs/home/bsc/bsc072871/repos/PLACER/","/shared/work/NBD_Utilities/PLACER/"}
        if PLACER_PATH not in valid_PLACER_PATH:
            raise ValueError(f"Invalid path! option. Choose from {valid_PLACER_PATH}")

        # Validate rerank option
        rerank_options = {"prmsd", "plddt_pde", "plddt"}
        if rerank not in rerank_options:
            raise ValueError(f"Invalid rerank option '{rerank}'. Choose from {rerank_options}")

        # Validate mutation options
        if mutate:
            if not isinstance(mutate, dict):
                raise TypeError("Expected 'mutate' to be a dictionary mapping model names to residue numbers.")
            if not mutate_to or not isinstance(mutate_to, str) or len(mutate_to) != 3:
                raise ValueError("Expected 'mutate_to' to be a 3-letter residue code string.")

        # Validate ligand options
        if ligand:
            if apo:
                raise ValueError("Cannot specify both ligand and apo at the same time!.")

        # Prepare output directories
        os.makedirs(PLACERfolder, exist_ok=True)
        input_pdbs_folder = os.path.join(PLACERfolder, "input_pdbs")
        os.makedirs(input_pdbs_folder, exist_ok=True)

        # Save input models
        self.saveModels(input_pdbs_folder)

        # Generate PLACER commands
        jobs = []
        for model in self:
            pdb_name = f"{model}.pdb"
            pdb_path = os.path.join(input_pdbs_folder, pdb_name)

            command = f"python {PLACER_PATH}run_PLACER.py "
            command +=  f"-f {pdb_path} "
            command +=  f"-o {PLACERfolder}/{output_folder} "
            command +=  f"--nsamples {num_samples} "

            if suffix:
                command += f"--suffix {suffix} "
            if rerank:
                command += f"--rerank {rerank} "
            if ligand:
                command += f"--predict_ligand {ligand} "
            if apo:
                command += f"--no-use_sm "
            if mutate:
                command += f"--mutate {mutate[model]}{mutate_chain}:{mutate_to} "
                command += "--no-use_sm "
            if residue_json:
                command += f"--residue_json {residue_json} "
            command += "\n"
            jobs.append(command)

        return jobs

    def analyseDocking(
        self,
        docking_folder,
        protein_atoms=None,
        angles=None,
        atom_pairs=None,
        skip_chains=False,
        return_failed=False,
        ignore_hydrogens=False,
        separator="-",
        overwrite=False,
        only_models=None,
        output_folder='.analysis',
    ):
        """
        Analyse a Glide Docking simulation. The function allows to calculate ligand
        distances with the options protein_atoms or protein_pairs. With the first option
        the analysis will calculate the closest distance between the protein atoms given
        and any ligand atom (or heavy atom if ignore_hydrogens=True). The analysis will
        also return which ligand atom is the closest for each pose. On the other hand, with
        the atom_pairs option only distances for the specific atom pairs between the
        protein and the ligand will be calculated.

        The protein_atoms dictionary must contain as keys the model names (see iterable of this class),
        and as values a list of tuples, with each tuple representing a protein atom:
            {model1_name: [(chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name), ...], model2_name:...}

        The atom pairs must be given in a dicionary with each key representing the name
        of a model and each value  a sub dicionary with the ligands as keys and a list of the atom pairs
        to calculate in the format:
            {model1_name: { ligand_name : [((chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name)), ...],...} model2_name:...}

        Paramaeters
        ===========
        docking_folder : str
            Path to the folder where the docking resuts are (the format comes from the setUpGlideDocking() function.
        protein_atoms : dict
            Protein atoms to use for the closest distance calculation.
        atom_pairs : dict
            Protein and ligand atoms to use for distances calculation.
        skip_chains : bool
            Consider chains when atom tuples are given?
        return_failed : bool
            Return failed dockings as a list?
        ignore_hydrogens : bool
            With this option ligand hydrogens will be ignored for the closest distance (i.e., protein_atoms) calculation.
        separator : str
            Symbol to use for separating protein from ligand names. Should not be found in any model or ligand name.
        overwrite : bool
            Rerun analysis.
        """

        # Create analysis folder
        if not os.path.exists(docking_folder + '/'+output_folder):
            os.mkdir(docking_folder + '/'+output_folder)

        # Create scores data folder
        if not os.path.exists(docking_folder + '/'+output_folder+"/scores"):
            os.mkdir(docking_folder + '/'+output_folder+"/scores")

        # Create distance data folder
        if not os.path.exists(docking_folder + '/'+output_folder+"/atom_pairs"):
            os.mkdir(docking_folder + '/'+output_folder+"/atom_pairs")

        # Create angle data folder
        if angles:
            if not os.path.exists(docking_folder + '/'+output_folder+"/angles"):
                os.mkdir(docking_folder + '/'+output_folder+"/angles")

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        prepare_proteins._copyScriptFile(
            docking_folder + '/'+output_folder, "analyse_docking.py"
        )
        script_path = docking_folder + '/'+output_folder+"/._analyse_docking.py"

        # Write protein_atoms dictionary to json file
        if protein_atoms:
            with open(docking_folder + '/'+output_folder+"/._protein_atoms.json", "w") as jf:
                json.dump(protein_atoms, jf)

        if isinstance(only_models, str):
            only_models = [only_models]

        # Write atom_pairs dictionary to json file
        if atom_pairs:
            with open(docking_folder + '/'+output_folder+"/._atom_pairs.json", "w") as jf:
                json.dump(atom_pairs, jf)

        # Write angles dictionary to json file
        if angles:
            with open(docking_folder + '/'+output_folder+"/._angles.json", "w") as jf:
                json.dump(angles, jf)

        command = (
            "run "
            + docking_folder
            + '/'+output_folder+"/._analyse_docking.py "
            + docking_folder
        )
        if atom_pairs:
            command += (
                " --atom_pairs " + docking_folder + '/'+output_folder+"/._atom_pairs.json"
            )
        elif protein_atoms:
            command += (
                " --protein_atoms " + docking_folder + '/'+output_folder+"/._protein_atoms.json"
            )
        if angles:
            command += " --angles " + docking_folder + '/'+output_folder+"/._angles.json"
        if skip_chains:
            command += " --skip_chains"
        if return_failed:
            command += " --return_failed"
        if ignore_hydrogens:
            command += " --ignore_hydrogens"
        command += " --separator " + separator
        if only_models:
            command += " --only_models " + ",".join(only_models)
        else:
            command += " --only_models " + ",".join(self.models_names)
        if overwrite:
            command += " --overwrite "
        command += ' --analysis_folder '+output_folder

        os.system(command)

        # # Read the CSV file into pandas
        # if not os.path.exists(docking_folder + '/'+output_folder+"/docking_data.csv"):
        #     raise ValueError(
        #         "Docking analysis failed. Check the ouput of the analyse_docking.py script."
        #     )

        # Read scores data
        scores_directory = docking_folder + '/'+output_folder+"/scores"
        self.docking_data = []
        for f in os.listdir(scores_directory):
            model = f.split(separator)[0]
            ligand = f.split(separator)[1].split(".")[0]

            # Read the CSV file into pandas
            self.docking_data.append(pd.read_csv(
                scores_directory+'/'+f
            ))

        # Concatenate the list of DataFrames into a single DataFrame
        self.docking_data = pd.concat(self.docking_data)
        self.docking_data.set_index(["Protein", "Ligand", "Pose"], inplace=True)

        distances_directory = docking_folder + '/'+output_folder+"/atom_pairs"
        for f in os.listdir(distances_directory):
            model = f.split(separator)[0]
            ligand = f.split(separator)[1].split(".")[0]

            # Read the CSV file into pandas
            self.docking_distances.setdefault(model, {})
            self.docking_distances[model][ligand] = pd.read_csv(
                distances_directory+'/'+f
            )
            self.docking_distances[model][ligand].set_index(
                ["Protein", "Ligand", "Pose"], inplace=True
            )

            self.docking_ligands.setdefault(model, [])
            if ligand not in self.docking_ligands[model]:
                self.docking_ligands[model].append(ligand)

        angles_directory = docking_folder + '/'+output_folder+"/angles"
        if os.path.exists(angles_directory):
            for f in os.listdir(angles_directory):
                model = f.split(separator)[0]
                ligand = f.split(separator)[1].split(".")[0]

                # Read the CSV file into pandas
                self.docking_angles.setdefault(model, {})
                self.docking_angles[model][ligand] = pd.read_csv(
                    angles_directory +'/'+ f
                )
                self.docking_angles[model][ligand].set_index(
                    ["Protein", "Ligand", "Pose"], inplace=True
                )

        if return_failed:
            with open(docking_folder + '/'+output_folder+"/._failed_dockings.json") as jifd:
                failed_dockings = json.load(jifd)
            return failed_dockings

    def analyseDockingParallel(self,
        docking_folder,
        protein_atoms=None,
        angles=None,
        atom_pairs=None,
        skip_chains=False,
        return_failed=False,
        ignore_hydrogens=False,
        separator="-",
        overwrite=False,
        only_models=None,
        compute_sasa=False,
        output_folder='.analysis'):
        """
        Set up jobs for analysing individual docking and creating CSV files. The files should be
        read by the analyseDocking function (i.e., the non-parallel version).
        """

        # Create analysis folder
        if not os.path.exists(docking_folder + '/'+output_folder):
            os.mkdir(docking_folder + '/'+output_folder)

        # Create scores data folder
        if not os.path.exists(docking_folder + '/'+output_folder+"/scores"):
            os.mkdir(docking_folder + '/'+output_folder+"/scores")

        # Create distance data folder
        if not os.path.exists(docking_folder + '/'+output_folder+"/atom_pairs"):
            os.mkdir(docking_folder + '/'+output_folder+"/atom_pairs")

        # Create angle data folder
        if angles:
            if not os.path.exists(docking_folder + '/'+output_folder+"/angles"):
                os.mkdir(docking_folder + '/'+output_folder+"/angles")

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        prepare_proteins._copyScriptFile(
            docking_folder + '/'+output_folder, "analyse_individual_docking.py"
        )
        script_path = docking_folder + '/'+output_folder+"/._analyse_individual_docking.py"

        # Write protein_atoms dictionary to json file
        if protein_atoms:
            with open(docking_folder + '/'+output_folder+"/._protein_atoms.json", "w") as jf:
                json.dump(protein_atoms, jf)

        if isinstance(only_models, str):
            only_models = [only_models]

        # Write atom_pairs dictionary to json file
        if atom_pairs:
            with open(docking_folder + '/'+output_folder+"/._atom_pairs.json", "w") as jf:
                json.dump(atom_pairs, jf)

        # Write angles dictionary to json file
        if angles:
            with open(docking_folder + '/'+output_folder+"/._angles.json", "w") as jf:
                json.dump(angles, jf)

        jobs = []
        for model in os.listdir(docking_folder+'/output_models'):

            # Skip models not given in only_models
            if only_models != None  and model not in only_models:
                continue

            # Check separator in model name
            if separator in model:
                raise ValueError('The separator %s was found in model name %s. Please use a different one!' % (separator, model))

            for f in os.listdir(docking_folder+'/output_models/'+model):

                subjobs = None
                mae_output = None

                if f.endswith('.maegz'):
                    ligand = f.replace(model+'_','').replace('_pv.maegz','')

                    # Check separator in ligand name
                    if separator in ligand:
                        raise ValueError('The separator %s was found in ligand name %s. Please use a different one!' % (separator, ligand))

                    mae_output = docking_folder+'/output_models/'+model+'/'+f

                    csv_name = model+separator+ligand+'.csv'
                    scores_csv = docking_folder+'/'+output_folder+'/scores/'+csv_name
                    distance_csv = docking_folder+'/'+output_folder+'/atom_pairs/'+csv_name
                    angles_csv = docking_folder+'/'+output_folder+'/angles/'+csv_name

                    skip_scores = True
                    skip_distances = True
                    skip_angles = True

                    if not os.path.exists(scores_csv) or overwrite:
                        skip_scores = False

                    if atom_pairs and not os.path.exists(distance_csv) or overwrite:
                        skip_distances = False

                    if angles and not os.path.exists(angles_csv) or overwrite:
                        skip_angles = False

                    if skip_scores and skip_distances and skip_angles:
                        continue

                    command  = 'run '
                    command += docking_folder+'/'+output_folder+"/._analyse_individual_docking.py "
                    command += docking_folder+' '
                    command += mae_output+' '
                    command += model+' '
                    command += ligand+' '
                    if atom_pairs:
                        command += "--atom_pairs " + docking_folder + '/'+output_folder+"/._atom_pairs.json "
                    elif protein_atoms:
                        command += "--protein_atoms " + docking_folder + '/'+output_folder+"/._protein_atoms.json "
                    if angles:
                        command += " --angles " + docking_folder + '/'+output_folder+"/._angles.json"
                    if skip_chains:
                        command += " --skip_chains"
                    if return_failed:
                        command += " --return_failed"
                    if ignore_hydrogens:
                        command += " --ignore_hydrogens"
                    if compute_sasa:
                        command += " --compute_sasa"
                    command += " --separator " + separator
                    command += '\n'
                    jobs.append(command)

        return jobs

    def analyseRosettaDocking(self, docking_folder, separator='-', only_models=None,
                              atom_pairs=None, energy_by_residue=False,
                              interacting_residues=False, query_residues=None,
                              overwrite=False, protonation_states=False,
                              binding_energy=False, decompose_bb_hb_into_pair_energies=False,
                              cpus=None, return_jobs=False, verbose=False,
                              skip_finished=False, pyrosetta_env=None, param_files=None):
        """
        Analyse the Rosetta docking folder. By default this function falls back to parsing
        the Rosetta score files so it works without PyRosetta, but when options such as
        `atom_pairs`, `energy_by_residue`, or `binding_energy` are requested it reuses
        the same PyRosetta-driven `analyse_calculation.py` script that powers
        `analyseRosettaCalculation`.

        Parameters
        ----------
        docking_folder : str
            Path to the folder where the Rosetta docking results are stored.
        separator : str, optional
            Symbol that separates the protein and ligand names, default is '_' and should
            not appear more than once in any folder name created by `setUpRosettaDocking`.
        only_models : str or Sequence[str], optional
            Limit the analysis to this list of protein models (ligand names are inferred
            from the folder name after the separator).
        atom_pairs : dict, optional
            Dictionary of atom pairs to measure. Keys may be either the combined folder
            names (Model{separator}Ligand) or the plain model names - keys matching the model
            will be expanded to all associated ligands. Each atom pair can be expressed in
            the legacy ((chain, residue, atom), (chain, residue, atom)) form, or you can
            provide ligand-side tuples like (('chain', res, atom), 'ATOM_NAME'); the ligand
            atom names are resolved automatically from the `ligand_params/<ligand>/<ligand>.pdb`
            files so you only need to supply the atom name on the ligand side.
        energy_by_residue : bool, optional
            If True, request the energy-by-residue analysis from the PyRosetta script.
        interacting_residues : bool, optional
            Compute neighbouring interaction energies.
        query_residues : list[int], optional
            Residue IDs to query when running interacting residue/protonation analyses.
        overwrite : bool, optional
            Remove existing analysis files before rerunning the PyRosetta script.
        protonation_states : bool, optional
            Gather protonation data via PyRosetta.
        binding_energy : str or bool, optional
            Chain names for which the binding energy is calculated (comma-separated
            string). If False, binding energies are skipped.
        decompose_bb_hb_into_pair_energies : bool, optional
            Pass the same flag to the PyRosetta analysis.
        cpus : int, optional
            Number of CPUs to feed to the PyRosetta script.
        return_jobs : bool, optional
            Return a list of shell commands instead of running the analysis; the commands
            will each target a single model/ligand folder and must be executed manually.
        verbose : bool, optional
            Increase verbosity of the PyRosetta script.
        skip_finished : bool, optional
            When `return_jobs` is True, omit commands for models that already have
            `.analysis/scores/<model>.csv` files.
        pyrosetta_env : str, optional
            Name of the conda environment to activate before running the PyRosetta analysis script.
        """

        output_models_path = os.path.join(docking_folder, 'output_models')
        if not os.path.exists(output_models_path):
            raise ValueError(f"Rosetta docking output folder '{output_models_path}' does not exist.")

        model_ligands = [d for d in os.listdir(output_models_path)
                         if os.path.isdir(os.path.join(output_models_path, d))]
        if only_models is not None:
            if isinstance(only_models, str):
                only_models = [only_models]
            model_ligands = [ml for ml in model_ligands if ml.split(separator)[0] in only_models]

        if not model_ligands:
            raise ValueError("No docking models found after applying filters.")

        for model_ligand in model_ligands:
            if model_ligand.count(separator) != 1:
                raise ValueError(
                    f"The separator '{separator}' was not found or was found more than once in '{model_ligand}'."
                )

        pyrosetta_features = any([
            atom_pairs,
            energy_by_residue,
            interacting_residues,
            protonation_states,
            binding_energy,
            decompose_bb_hb_into_pair_energies,
            return_jobs,
        ])

        if not pyrosetta_features:
            # Fall back to parsing the raw SCORE files.
            self.rosetta_docking = pd.DataFrame()
            self.rosetta_docking_distances = {}
            total_files = len(model_ligands)
            processed_files = 0

            for model_ligand in model_ligands:
                model, ligand = model_ligand.split(separator)
                try:
                    scorefile_out = os.path.join(output_models_path, f"{model_ligand}/{model_ligand}.out")
                    scorefile_sc = os.path.join(output_models_path, f"{model_ligand}/{model_ligand}.sc")

                    if os.path.exists(scorefile_out):
                        scorefile = scorefile_out
                    elif os.path.exists(scorefile_sc):
                        scorefile = scorefile_sc
                    else:
                        raise FileNotFoundError(
                            f"Neither '{model_ligand}.out' nor '{model_ligand}.sc' was found."
                        )

                    scores = _readRosettaScoreFile(scorefile)
                    scores['Ligand'] = ligand
                    scores = scores.set_index(['Model', 'Ligand', 'Pose'])

                    distance_columns = [col for col in scores.columns if col.startswith('distance_')]
                    distance_df = scores[distance_columns]

                    self.rosetta_docking_distances.setdefault(model, {})[ligand] = distance_df
                    scores = scores.drop(columns=distance_columns)

                    interface_delta_col = next((col for col in scores.columns if col.startswith('interface_delta_')), None)
                    if interface_delta_col and 'total_score' in scores.columns:
                        cols = list(scores.columns)
                        cols.insert(cols.index('total_score') + 1, cols.pop(cols.index(interface_delta_col)))
                        scores = scores[cols]

                    self.rosetta_docking = pd.concat([self.rosetta_docking, scores])

                except FileNotFoundError as e:
                    print(f"\nSkipping {model_ligand} due to missing file: {e}")
                    continue

                processed_files += 1
                progress = f"Processing: {processed_files}/{total_files} files"
                sys.stdout.write('\r' + progress)
                sys.stdout.flush()

            print()
            self.rosetta_docking_data = self.rosetta_docking
            return self.rosetta_docking

        # --- PyRosetta-driven analysis path ---
        analysis_folder = os.path.join(docking_folder, '.analysis')
        os.makedirs(analysis_folder, exist_ok=True)
        _copyScriptFile(docking_folder, "analyse_calculation.py", subfolder="pyrosetta")
        script_path = os.path.join(docking_folder, '._analyse_calculation.py')
        distances_folder = os.path.join(analysis_folder, 'distances')

        expanded_atom_pairs = None
        missing_ligand_atoms = {}
        models_without_pairs = []
        if atom_pairs is not None:
            ligand_chain_overrides = _collect_ligand_chain_overrides(docking_folder, model_ligands, separator)
            expanded_atom_pairs, missing_ligand_atoms, models_without_pairs = _prepare_expanded_atom_pairs(
                atom_pairs, docking_folder, model_ligands, separator, ligand_chain_overrides=ligand_chain_overrides
            )
            if not expanded_atom_pairs:
                requested_pairs = _collect_requested_model_ligand_pairs(atom_pairs)
                requested_ligands = _collect_requested_ligands(atom_pairs)
                if not requested_ligands:
                    fallback = []
                    for ml in model_ligands:
                        if separator in ml:
                            fallback.append(ml.rsplit(separator, 1)[-1])
                        else:
                            fallback.append(ml)
                    requested_ligands = set(fallback)
                ligand_atom_summaries = []
                for ligand in sorted(requested_ligands):
                    pdb_path = os.path.join(
                        docking_folder, "ligand_params", ligand, f"{ligand}.pdb"
                    )
                    atoms = _list_ligand_atom_names(docking_folder, ligand)
                    if atoms:
                        suffix = "..." if len(atoms) > 10 else ""
                        ligand_atom_summaries.append(
                            f"{ligand}: {', '.join(atoms[:10])}{suffix} (from {pdb_path})"
                        )
                        continue

                    params_atoms, params_path = _list_params_atom_names(docking_folder, ligand)
                    if params_atoms:
                        suffix = "..." if len(params_atoms) > 10 else ""
                        ligand_atom_summaries.append(
                            f"{ligand}: {', '.join(params_atoms[:10])}{suffix} (from {params_path})"
                        )
                        continue

                    paths_checked = [p for p in (pdb_path, params_path) if p]
                    path_note = (
                        f" Checked files: {', '.join(paths_checked)}."
                        if paths_checked
                        else ""
                    )
                    ligand_atom_summaries.append(
                        f"{ligand}: (no atoms found in ligand_params/{ligand}/{ligand}.pdb;{path_note})"
                    )
                detail = "\n".join(ligand_atom_summaries) if ligand_atom_summaries else "no ligands could be derived from the selected models."
                if missing_ligand_atoms:
                    missing_lines = []
                    for (model, ligand), atoms in sorted(missing_ligand_atoms.items()):
                        atom_list = ", ".join(sorted(atoms))
                        missing_lines.append(f"{model}/{ligand}: missing requested atoms: {atom_list}")
                    detail += "\nMissing requested ligand atoms:\n" + "\n".join(missing_lines)
                if models_without_pairs:
                    detail += (
                        "\nNo atom pair entries were matched for the following docking models: "
                        + ", ".join(sorted(models_without_pairs))
                    )
                if requested_pairs:
                    unmatched = []
                    for model, ligand in sorted(requested_pairs):
                        candidate = f"{model}{separator}{ligand}"
                        if candidate not in expanded_atom_pairs:
                            unmatched.append(candidate)
                    if unmatched:
                        detail += (
                            "\nRequested model/ligand pairs that were never considered: "
                            + ", ".join(unmatched)
                        )
                raise ValueError(
                    "No atom pairs matched the selected docking models. "
                    "The `atom_pairs` dictionary must follow the {model: {ligand: [(protein_atom, ligand_atom), ...]}} "
                    "structure rather than embedding both names in a single key. "
                    "Ligand atom names are listed below:\n"
                    f"{detail}"
                )
        atom_pairs_file = (
            os.path.join(docking_folder, '._atom_pairs.json')
            if expanded_atom_pairs
            else None
        )
        if atom_pairs_file:
            with open(atom_pairs_file, 'w') as jf:
                json.dump(expanded_atom_pairs, jf)

        if overwrite:
            for subfolder in ['scores', 'distances', 'binding_energy', 'ebr', 'neighbours', 'protonation']:
                folder = os.path.join(analysis_folder, subfolder)
                if os.path.isdir(folder):
                    for model_ligand in model_ligands:
                        path = os.path.join(folder, f"{model_ligand}.csv")
                        if os.path.exists(path):
                            os.remove(path)

        os.makedirs(os.path.join(analysis_folder, 'scores'), exist_ok=True)
        if expanded_atom_pairs is not None:
            os.makedirs(os.path.join(analysis_folder, 'distances'), exist_ok=True)
        if binding_energy:
            os.makedirs(os.path.join(analysis_folder, 'binding_energy'), exist_ok=True)
        if energy_by_residue:
            os.makedirs(os.path.join(analysis_folder, 'ebr'), exist_ok=True)
        if interacting_residues:
            os.makedirs(os.path.join(analysis_folder, 'neighbours'), exist_ok=True)
        if protonation_states:
            os.makedirs(os.path.join(analysis_folder, 'protonation'), exist_ok=True)

        command = f"python {script_path} {docking_folder} "
        if binding_energy:
            command += f"--binding_energy {binding_energy} "
        if atom_pairs_file:
            command += f"--atom_pairs {atom_pairs_file} "
        if energy_by_residue:
            command += "--energy_by_residue "
        if interacting_residues:
            command += "--interacting_residues "
            if query_residues is not None:
                command += "--query_residues " + ",".join([str(r) for r in query_residues]) + " "
        if protonation_states:
            command += "--protonation_states "
        if decompose_bb_hb_into_pair_energies:
            command += "--decompose_bb_hb_into_pair_energies "
        if cpus is not None:
            command += f"--cpus {cpus} "
        if verbose:
            command += "--verbose "
        if overwrite:
            command += "--overwrite "

        if return_jobs:
            command += "--models MODEL "
        else:
            command += "--models " + ",".join(model_ligands) + " "

        if return_jobs:
            commands = []
            for model_ligand in model_ligands:
                score_file = os.path.join(analysis_folder, 'scores', f"{model_ligand}.csv")
                if skip_finished and os.path.exists(score_file):
                    continue
                job_command = command.replace('MODEL', model_ligand)
                if pyrosetta_env:
                    job_command = _wrap_pyrosetta_command(job_command, pyrosetta_env)
                commands.append(job_command)

            print("Returning jobs for running the analysis in parallel.")
            print("After jobs have finished, rerun this function removing return_jobs=True!")
            return commands

        missing = False
        for model_ligand in model_ligands:
            score_file = os.path.join(analysis_folder, 'scores', f"{model_ligand}.csv")
            if not os.path.exists(score_file):
                missing = True
                break

        if missing:
            if not pyrosetta_env:
                installed = {pkg.key for pkg in pkg_resources.working_set}
                if 'pyrosetta' not in installed:
                    raise ValueError(
                        'PyRosetta was not found in your Python environment. '
                        'Consider using return_jobs=True or activating an environment that does have it.'
                    )
            exec_command = command
            if pyrosetta_env:
                exec_command = _wrap_pyrosetta_command(exec_command, pyrosetta_env)
            os.system(exec_command)

        self.rosetta_docking = pd.DataFrame()
        self.rosetta_docking_distances = {}
        total_files = len(model_ligands)
        processed_files = 0

        for model_ligand in model_ligands:
            score_file = os.path.join(analysis_folder, 'scores', f"{model_ligand}.csv")
            if not os.path.exists(score_file):
                print(f"Score file for {model_ligand} not found, skipping.")
                continue

            scores = pd.read_csv(score_file)
            base_model, ligand = model_ligand.split(separator, 1)
            scores['Model'] = base_model
            scores['Ligand'] = ligand
            scores = scores.set_index(['Model', 'Ligand', 'Pose'])

            distance_columns = [col for col in scores.columns if col.startswith('distance_')]
            distance_df = pd.DataFrame()
            if distance_columns:
                distance_df = scores[distance_columns]
                scores = scores.drop(columns=distance_columns)
                if not distance_df.empty:
                    distance_df = distance_df.sort_index(level=['Model', 'Ligand', 'Pose'])
            elif atom_pairs:
                distance_file = os.path.join(distances_folder, f"{model_ligand}.csv")
                if os.path.exists(distance_file):
                    raw_distances = pd.read_csv(distance_file)
                    if {'Model', 'Pose'}.issubset(raw_distances.columns):
                        filtered = raw_distances.copy()
                        mask = filtered['Model'] == model_ligand
                        if mask.any():
                            filtered = filtered[mask]
                        filtered['Ligand'] = ligand
                        filtered['Model'] = base_model
                        distance_df = filtered.set_index(['Model', 'Ligand', 'Pose']).sort_index(level=['Model', 'Ligand', 'Pose'])

            if not distance_df.empty:
                self.rosetta_docking_distances.setdefault(base_model, {})[ligand] = distance_df

            interface_delta_col = next((col for col in scores.columns if col.startswith('interface_delta_')), None)
            if interface_delta_col and 'total_score' in scores.columns:
                cols = list(scores.columns)
                cols.insert(cols.index('total_score') + 1, cols.pop(cols.index(interface_delta_col)))
                scores = scores[cols]

            if self.rosetta_docking.empty:
                self.rosetta_docking = scores
            else:
                self.rosetta_docking = pd.concat([self.rosetta_docking, scores])

            processed_files += 1
            progress = f"Processing: {processed_files}/{total_files} files"
            sys.stdout.write('\r' + progress)
            sys.stdout.flush()

        print()
        self.rosetta_docking_data = self.rosetta_docking
        return self.rosetta_docking

    def _set_rosetta_metric_type(self, metric_name: str, metric_type: str):
        """
        Record the type ('distance', 'angle', etc.) for a Rosetta docking metric,
        storing both prefixed (metric_*) and bare names for convenience.
        """
        if not hasattr(self, 'rosetta_docking_metric_type') or self.rosetta_docking_metric_type is None:
            self.rosetta_docking_metric_type = {}
        prefixed = metric_name if metric_name.startswith('metric_') else f"metric_{metric_name}"
        bare = prefixed[7:] if prefixed.startswith('metric_') else prefixed
        self.rosetta_docking_metric_type[prefixed] = metric_type
        self.rosetta_docking_metric_type[bare] = metric_type

    def _get_rosetta_metric_type(self, metric_name: str):
        """
        Retrieve the recorded metric type, accepting either prefixed or bare names.
        """
        if not hasattr(self, 'rosetta_docking_metric_type') or self.rosetta_docking_metric_type is None:
            self.rosetta_docking_metric_type = {}
        prefixed = metric_name if metric_name.startswith('metric_') else f"metric_{metric_name}"
        bare = prefixed[7:] if prefixed.startswith('metric_') else prefixed
        if prefixed in self.rosetta_docking_metric_type:
            return self.rosetta_docking_metric_type[prefixed]
        return self.rosetta_docking_metric_type.get(bare)

    def combineRosettaDockingDistancesIntoMetrics(self, catalytic_labels, overwrite=False):
        """
        Combine different equivalent distances into specific named metrics. The function
        takes as input a dictionary (catalytic_labels) composed of inner dictionaries as follows:

            catalytic_labels = {
                metric_name = {
                    model = {
                        ligand = distances_list}}}

        The innermost distances_list object contains all equivalent distance names for
        a specific model and ligand pair to be combined under the same metric_name column.

        The combination is done by taking the minimum value of all equivalent distances.

        Parameters
        ==========
        catalytic_labels : dict
            Dictionary defining which distances will be combined under a common name.
            (for details see above).
        """

        # Initialize the metric type dictionary if it doesn't exist
        if not hasattr(self, 'rosetta_docking_metric_type'):
            self.rosetta_docking_metric_type = {}

        for name in catalytic_labels:
            if "metric_" + name in self.rosetta_docking.columns and not overwrite:
                print(
                    f"Combined metric {name} already added. Give overwrite=True to recombine"
                )
            else:
                values = []
                for model in self.rosetta_docking.index.get_level_values('Model').unique():

                    # Check whether model is found in docking distances
                    if model not in self.rosetta_docking_distances:
                        continue

                    model_series = self.rosetta_docking[
                        self.rosetta_docking.index.get_level_values("Model") == model
                    ]

                    for ligand in self.rosetta_docking.index.get_level_values('Ligand').unique():

                        # Check whether ligand is found in model's docking distances
                        if ligand not in self.rosetta_docking_distances[model]:
                            continue

                        ligand_series = model_series[
                            model_series.index.get_level_values("Ligand") == ligand
                        ]

                        # Check input metric
                        distance_metric = False
                        angle_metric = False
                        for x in catalytic_labels[name][model][ligand]:
                            if len(x.split("-")) == 2:
                                distance_metric = True
                            elif len(x.split("-")) == 3:
                                angle_metric = True

                        if distance_metric and angle_metric:
                            raise ValueError(
                                f"Metric {name} combines distances and angles which is not supported."
                            )

                        if distance_metric:
                            distances = catalytic_labels[name][model][ligand]
                            distance_values = (
                                self.rosetta_docking_distances[model][ligand][distances]
                                .min(axis=1)
                                .tolist()
                            )
                            assert ligand_series.shape[0] == len(distance_values)
                            values += distance_values
                            self._set_rosetta_metric_type(name, "distance")
                        elif angle_metric:
                            angles = catalytic_labels[name][model][ligand]
                            if len(angles) > 1:
                                raise ValueError(
                                    "Combining more than one angle into a metric is not currently supported."
                                )
                            angle_values = (
                                self.rosetta_docking_angles[model][ligand][angles]
                                .min(axis=1)
                                .tolist()
                            )
                            assert ligand_series.shape[0] == len(angle_values)
                            values += angle_values
                            self._set_rosetta_metric_type(name, "angle")

                self.rosetta_docking["metric_" + name] = values

    def combineRosettaDockingMetricsWithExclusions(self, combinations, exclusions, drop=True):
        """
        Combine mutually exclusive Rosetta docking metrics into new metrics while
        respecting exclusion rules.

        Parameters
        ----------
        combinations : dict
            Dictionary defining which metrics to combine under a new common name, e.g.:
                {
                    "Catalytic": ("MetricA", "MetricB"),
                    ...
                }
            Metric names should be provided without the leading ``metric_`` prefix.

        exclusions : list or dict
            Either a list of tuples describing mutually exclusive metrics or a
            dictionary mapping a metric to the metrics it should exclude when it is
            selected as the minimum.

        drop : bool, optional
            If True, drop the original metric columns after combining. Default True.
        """

        if self.rosetta_docking_data is None or self.rosetta_docking_data.empty:
            raise ValueError("Rosetta docking data is empty. Run analyseRosettaDocking first.")

        if not hasattr(self, 'rosetta_docking_metric_type'):
            self.rosetta_docking_metric_type = {}

        def _with_prefix(name: str) -> str:
            return name if name.startswith('metric_') else f"metric_{name}"

        def _strip_prefix(name: str) -> str:
            return name[7:] if name.startswith('metric_') else name

        # Determine exclusion type
        if isinstance(exclusions, list):
            simple_exclusions = True
            by_metric_exclusions = False
        elif isinstance(exclusions, dict):
            simple_exclusions = False
            by_metric_exclusions = True
        else:
            raise ValueError('exclusions should be a list of tuples or a dictionary by metrics.')

        # Collect all unique metrics involved
        unique_metrics = set()
        for new_metric, metrics in combinations.items():
            metric_types = []
            for metric in metrics:
                metric_label = _with_prefix(metric)
                if metric_label not in self.rosetta_docking_metric_type:
                    raise ValueError(f"Metric '{metric}' is not present in rosetta_docking_metric_type.")
                metric_types.append(self.rosetta_docking_metric_type[metric_label])
            if len(set(metric_types)) != 1:
                raise ValueError('Attempting to combine different metric types (e.g., distances and angles) is not allowed.')
            self._set_rosetta_metric_type(new_metric, metric_types[0])
            unique_metrics.update(_strip_prefix(metric) for metric in metrics)

        metrics_list = list(unique_metrics)
        metrics_indexes = {m: idx for idx, m in enumerate(metrics_list)}

        # Add metric prefix if not already present
        add_metric_prefix = True
        for m in metrics_list:
            if 'metric_' in m:
                raise ValueError('Provide metric names without the "metric_" prefix.')
        all_metrics_columns = ['metric_' + m for m in metrics_list]

        missing_columns = set(all_metrics_columns) - set(self.rosetta_docking_data.columns)
        if missing_columns:
            raise ValueError(f"Missing metric columns in data: {missing_columns}")

        data = self.rosetta_docking_data[all_metrics_columns]
        excluded_positions = set()
        min_metric_labels = data.idxmin(axis=1)

        if simple_exclusions:
            for row_idx, metric_col_label in enumerate(min_metric_labels):
                m = metric_col_label.replace('metric_', '')

                for exclusion_group in exclusions:
                    canonical_group = {_strip_prefix(x) for x in exclusion_group}
                    if m in canonical_group:
                        others = canonical_group - {m}
                        for x in others:
                            if x in metrics_indexes:
                                col_idx = metrics_indexes[x]
                                excluded_positions.add((row_idx, col_idx))

                for metrics_group in combinations.values():
                    canonical_group = [_strip_prefix(x) for x in metrics_group]
                    if m in canonical_group:
                        others = set(canonical_group) - {m}
                        for y in others:
                            if y in metrics_indexes:
                                col_idx = metrics_indexes[y]
                                excluded_positions.add((row_idx, col_idx))

        data_array = data.to_numpy()

        if by_metric_exclusions:
            exclusions_map = {
                _strip_prefix(metric): [_strip_prefix(x) for x in excluded]
                for metric, excluded in exclusions.items()
            }

            for row_idx in range(data_array.shape[0]):
                considered_metrics = set()

                while True:
                    min_value = np.inf
                    min_col_idx = -1

                    for col_idx, metric_value in enumerate(data_array[row_idx]):
                        if col_idx not in considered_metrics and (row_idx, col_idx) not in excluded_positions:
                            if metric_value < min_value:
                                min_value = metric_value
                                min_col_idx = col_idx

                    if min_col_idx == -1:
                        break

                    considered_metrics.add(min_col_idx)

                    min_metric_label = data.columns[min_col_idx]
                    min_metric_name = _strip_prefix(min_metric_label)
                    excluded_metrics = exclusions_map.get(min_metric_name, [])

                    for excluded_metric in excluded_metrics:
                        if excluded_metric in metrics_indexes:
                            excluded_col_idx = metrics_indexes[excluded_metric]
                            if (row_idx, excluded_col_idx) not in excluded_positions:
                                excluded_positions.add((row_idx, excluded_col_idx))
                                data_array[row_idx, excluded_col_idx] = np.inf

        for new_metric_name, metrics_to_combine in combinations.items():
            canonical_metrics = [_strip_prefix(m) for m in metrics_to_combine]
            c_indexes = [metrics_indexes[m] for m in canonical_metrics if m in metrics_indexes]

            if c_indexes:
                combined_min = np.min(data_array[:, c_indexes], axis=1)

                if np.all(np.isinf(combined_min)):
                    print(f"Skipping combination for '{new_metric_name}' due to incompatible exclusions.")
                    continue
                self.rosetta_docking_data['metric_' + new_metric_name] = combined_min
            else:
                raise ValueError(f"No valid metrics to combine for '{new_metric_name}'.")

        if drop:
            self.rosetta_docking_data.drop(columns=all_metrics_columns, inplace=True)

        for new_metric_name, metrics_to_combine in combinations.items():
            non_excluded_found = False
            canonical_metrics = [_strip_prefix(m) for m in metrics_to_combine]

            for metric in canonical_metrics:
                col_idx = metrics_indexes.get(metric)
                if col_idx is not None:
                    column_values = data_array[:, col_idx]
                    if not np.all(np.isinf(column_values)):
                        non_excluded_found = True
                        break

            if not non_excluded_found:
                print(f"Warning: No non-excluded metrics available to combine for '{new_metric_name}'.")

    def rosettaDockingBindingEnergyLandscape(self, initial_threshold=3.5, vertical_line=None, xlim=None, ylim=None, clim=None, color=None,
                                             size=1.0, alpha=0.05, vertical_line_width=0.5, vertical_line_color='k', dataframe=None,
                                             title=None, no_xticks=False, no_yticks=False, no_xlabel=False, no_ylabel=False,
                                             no_cbar=False, xlabel=None, ylabel=None, clabel=None, relative_total_energy=False):
        """
        Plot binding energy as an interactive plot.

        Parameters
        ==========
        initial_threshold : float, optional
            Initial threshold value for metrics sliders. Default is 3.5.
        vertical_line : float, optional
            Position to plot a vertical line.
        xlim : tuple, optional
            The limits for the x-axis range.
        ylim : tuple, optional
            The limits for the y-axis range.
        clim : tuple, optional
            The limits for the color range.
        color : str, optional
            Column name to use for coloring the plot. Can also be a fixed color.
        size : float, optional
            Scale factor for the plot size. Default is 1.0.
        alpha : float, optional
            Alpha value for the scatter plot markers. Default is 0.05.
        vertical_line_width : float, optional
            Width of the vertical line. Default is 0.5.
        vertical_line_color : str, optional
            Color of the vertical line. Default is 'k' (black).
        dataframe : pandas.DataFrame, optional
            Dataframe containing the data. If not provided, self.rosetta_docking is used.
        title : str, optional
            Title of the plot.
        no_xticks : bool, optional
            If True, x-axis ticks are not shown. Default is False.
        no_yticks : bool, optional
            If True, y-axis ticks are not shown. Default is False.
        no_xlabel : bool, optional
            If True, the x-axis label is not shown. Default is False.
        no_ylabel : bool, optional
            If True, the y-axis label is not shown. Default is False.
        no_cbar : bool, optional
            If True, the color bar is not shown. Default is False.
        xlabel : str, optional
            Label for the x-axis. If not provided, defaults to the x parameter.
        ylabel : str, optional
            Label for the y-axis. If not provided, defaults to the y parameter.
        clabel : str, optional
            Label for the color bar. If not provided, defaults to color_column.
        relative_total_energy : bool, optional
            If True, color values are shown relative to their minimum value. Default is False.
        """

        if not self.rosetta_docking_distances:
            raise ValueError('There are no distances in the docking data. Use calculateDistances to show plot.')

        def getLigands(model, dataframe=None):
            if dataframe is not None:
                model_series = dataframe[dataframe.index.get_level_values('Model') == model]
            else:
                model_series = self.rosetta_docking[self.rosetta_docking.index.get_level_values('Model') == model]

            ligands = list(set(model_series.index.get_level_values('Ligand').tolist()))
            ligands_ddm = Dropdown(options=ligands, description='Ligand', style={'description_width': 'initial'})

            interact(getDistance, model_series=fixed(model_series), model=fixed(model), ligand=ligands_ddm)

        def getDistance(model_series, model, ligand, by_metric=False):
            ligand_series = model_series[model_series.index.get_level_values('Ligand') == ligand]

            distances = []
            distance_label = 'Distance'
            if by_metric:
                distances = [d for d in ligand_series if d.startswith('metric_') and not ligand_series[d].dropna().empty]
                distance_label = 'Metric'

            if not distances:
                if model in self.rosetta_docking_distances and ligand in self.rosetta_docking_distances[model]:
                    distances = [d for d in self.rosetta_docking_distances[model][ligand] if 'distance' in d or '_coordinate' in d]
                if model in self.rosetta_docking_angles and ligand in self.rosetta_docking_angles[model]:
                    distances += [d for d in self.rosetta_docking_angles[model][ligand] if 'angle' in d]
                if 'Ligand RMSD' in self.rosetta_docking:
                    distances.append('Ligand RMSD')

            if not distances:
                raise ValueError('No distances or metrics found for this ligand. Consider calculating some distances.')

            distances_ddm = Dropdown(options=distances, description=distance_label, style={'description_width': 'initial'})

            interact(getMetrics, distances=fixed(distances_ddm), ligand_series=fixed(ligand_series),
                     model=fixed(model), ligand=fixed(ligand))

        def getMetrics(ligand_series, distances, model, ligand, filter_by_metric=False, filter_by_label=False,
                       color_by_metric=False, color_by_labels=False):

            if color_by_metric or filter_by_metric:
                metrics = [k for k in ligand_series.keys() if 'metric_' in k]
                metrics_sliders = {}
                for m in metrics:
                    if self.rosetta_docking_metric_type[m] == 'distance':
                        m_slider = FloatSlider(value=initial_threshold, min=0, max=max(30, max(ligand_series[m])), step=0.1,
                                               description=f"{m}:", disabled=False, continuous_update=False,
                                               orientation='horizontal', readout=True, readout_format='.2f')
                    elif self.rosetta_docking_metric_type[m] in ['angle', 'torsion']:
                        m_slider = FloatRangeSlider(value=[110, 130], min=-180, max=180, step=0.1,
                                                    description=f"{m}:", disabled=False, continuous_update=False,
                                                    orientation='horizontal', readout=True, readout_format='.2f')

                    metrics_sliders[m] = m_slider
            else:
                metrics_sliders = {}

            if filter_by_label:
                labels_ddms = {}
                labels = [l for l in ligand_series.keys() if 'label_' in l]
                for l in labels:
                    label_options = [None] + sorted(list(set(ligand_series[l])))
                    labels_ddms[l] = Dropdown(options=label_options, description=l, style={'description_width': 'initial'})
            else:
                labels_ddms = {}

            interact(getColor, distance=distances, model=fixed(model), ligand=fixed(ligand),
                     metrics=fixed(metrics_sliders), ligand_series=fixed(ligand_series),
                     color_by_metric=fixed(color_by_metric), color_by_labels=fixed(color_by_labels), **labels_ddms)

        def getColor(distance, ligand_series, metrics, model, ligand, color_by_metric=False,
                     color_by_labels=False, **labels):

            if color is None:
                color_columns = [k for k in ligand_series.keys() if ':' not in k and 'distance' not in k and not k.startswith('metric_') and not k.startswith('label_')]
                color_columns = [None, 'Epoch'] + color_columns

                if 'interface_delta_B' in ligand_series:
                    be_column = 'interface_delta_B'
                else:
                    raise ValueError('No binding energy column (interface_delta_B) found in the data.')

                color_columns.remove(be_column)

                color_ddm = Dropdown(options=color_columns, description='Color', style={'description_width': 'initial'})
                if color_by_metric:
                    color_ddm.options = ['Color by metrics']
                    alpha_value = 0.10
                elif color_by_labels:
                    color_ddm.options = ['Color by labels']
                    alpha_value = 1.00
                else:
                    alpha_value = fixed(0.10)

                color_object = color_ddm
            else:
                color_object = fixed(color)

            interact(_bindingEnergyLandscape, color=color_object, ligand_series=fixed(ligand_series),
                     distance=fixed(distance), color_by_metric=fixed(color_by_metric), color_by_labels=fixed(color_by_labels),
                     Alpha=alpha_value, labels=fixed(labels), model=fixed(model), ligand=fixed(ligand), title=fixed(title),
                     no_xticks=fixed(no_xticks), no_yticks=fixed(no_yticks), no_cbar=fixed(no_cbar), clabel=fixed(clabel),
                     no_xlabel=fixed(no_xlabel), no_ylabel=fixed(no_ylabel), xlabel=fixed(xlabel), ylabel=fixed(ylabel),
                     relative_total_energy=fixed(relative_total_energy), clim=fixed(clim), **metrics)

        def _bindingEnergyLandscape(color, ligand_series, distance, model, ligand,
                                    color_by_metric=False, color_by_labels=False,
                                    Alpha=0.10, labels=None, title=None, no_xticks=False,
                                    no_yticks=False, no_cbar=False, no_xlabel=True, no_ylabel=False,
                                    xlabel=None, ylabel=None, clabel=None, relative_total_energy=False,
                                    clim=None, **metrics):

            skip_fp = False
            show_plot = True

            return_axis = False
            if color_by_metric:
                color = 'k'
                color_metrics = metrics
                metrics = {}
                return_axis = True
                show_plot = False

            elif color_by_labels:
                skip_fp = True
                return_axis = True
                show_plot = False

            if color == 'Total Energy' and relative_total_energy:
                relative_color_values = True
                if clim is None:
                    clim = (0, 27.631021116)
            else:
                relative_color_values = None

            if 'interface_delta_B' in ligand_series:
                be_column = 'interface_delta_B'
            else:
                raise ValueError('No binding energy column (interface_delta_B) found in the data.')

            if not skip_fp:
                axis = self.scatterPlotIndividualSimulation(model, ligand, distance, be_column, xlim=xlim, ylim=ylim,
                                                            vertical_line=vertical_line, color_column=color, clim=clim, size=size,
                                                            vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                            metrics=metrics, labels=labels, return_axis=return_axis, show=show_plot,
                                                            title=title, no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar,
                                                            no_xlabel=no_xlabel, no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel,
                                                            clabel=clabel, relative_color_values=relative_color_values, dataframe=ligand_series)

                # Set reasonable ticks
                if axis is not None:
                    if not no_xticks:
                        axis.set_xticks(axis.get_xticks()[::max(1, len(axis.get_xticks()) // 10 + 1)])
                    if not no_yticks:
                        axis.set_yticks(axis.get_yticks()[::max(1, len(axis.get_yticks()) // 10 + 1)])

            if color_by_metric:
                self.scatterPlotIndividualSimulation(model, ligand, distance, be_column, xlim=xlim, ylim=ylim,
                                                     vertical_line=vertical_line, color_column='r', clim=clim, size=size,
                                                     vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                     metrics=color_metrics, labels=labels, axis=axis, show=True, alpha=Alpha,
                                                     no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar, no_xlabel=no_xlabel,
                                                     no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel, clabel=clabel, dataframe=ligand_series)
            elif color_by_labels:
                all_labels = {l: sorted(list(set(ligand_series[l].to_list()))) for l in ligand_series.keys() if 'label_' in l}

                for l in all_labels:
                    colors = iter([plt.cm.Set2(i) for i in range(len(all_labels[l]))])
                    for i, v in enumerate(all_labels[l]):
                        if i == 0:
                            axis = self.scatterPlotIndividualSimulation(model, ligand, distance, be_column, xlim=xlim, ylim=ylim, plot_label=v,
                                                                        vertical_line=vertical_line, color_column=[next(colors)], clim=clim, size=size,
                                                                        vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                                        metrics=metrics, labels=labels, return_axis=return_axis, alpha=Alpha, show=show_plot,
                                                                        no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar, no_xlabel=no_xlabel,
                                                                        no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel, clabel=clabel, dataframe=ligand_series)
                            continue
                        elif i == len(all_labels[l]) - 1:
                            show_plot = True
                        axis = self.scatterPlotIndividualSimulation(model, ligand, distance, be_column, xlim=xlim, ylim=ylim, plot_label=v,
                                                                    vertical_line=vertical_line, color_column=[next(colors)], clim=clim, size=size,
                                                                    vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                                    metrics=metrics, labels={l: v}, return_axis=return_axis, axis=axis, alpha=Alpha, show=show_plot,
                                                                    show_legend=True, title=title, no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar,
                                                                    no_xlabel=no_xlabel, no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel, clabel=clabel,
                                                                    dataframe=ligand_series)

        models = self.rosetta_docking.index.get_level_values('Model').unique()
        models_ddm = Dropdown(options=models, description='Model', style={'description_width': 'initial'})

        interact(getLigands, model=models_ddm, dataframe=fixed(dataframe))

    def scatterPlotIndividualSimulation(self, model, ligand, x, y, vertical_line=None, color_column=None, size=1.0, labels_size=10.0, plot_label=None,
                                        xlim=None, ylim=None, metrics=None, labels=None, title=None, title_size=14.0, return_axis=False, dpi=300, show_legend=False,
                                        axis=None, xlabel=None, ylabel=None, vertical_line_color='k', vertical_line_width=0.5, marker_size=0.8, clim=None, show=False,
                                        clabel=None, legend_font_size=6, no_xticks=False, no_yticks=False, no_cbar=False, no_xlabel=False, no_ylabel=False,
                                        relative_color_values=False, dataframe=None, separator='_', **kwargs):
        """
        Creates a scatter plot for the selected model and ligand using the x and y
        columns. Data series can be filtered by specific metrics.

        Parameters
        ==========
        model : str
            The target model.
        ligand : str
            The target ligand.
        x : str
            The column name of the data to plot on the x-axis.
        y : str
            The column name of the data to plot on the y-axis.
        vertical_line : float, optional
            Position to plot a vertical line.
        color_column : str, optional
            The column name to use for coloring the plot. Also, a color can be given
            to use uniformly for the points.
        size : float, optional
            Scale factor for the plot size. Default is 1.0.
        labels_size : float, optional
            Font size for the labels. Default is 10.0.
        plot_label : str, optional
            Label for the plot. If not provided, it defaults to 'model_separator_ligand'.
        xlim : tuple, optional
            The limits for the x-axis range.
        ylim : tuple, optional
            The limits for the y-axis range.
        clim : tuple, optional
            The limits for the color range.
        metrics : dict, optional
            A set of metrics for filtering the data points.
        labels : dict, optional
            Use the label column values to filter the data.
        title : str, optional
            The title of the plot.
        title_size : float, optional
            Font size for the title. Default is 14.0.
        return_axis : bool, optional
            Whether to return the axis of this plot. Default is False.
        dpi : int, optional
            Dots per inch for the figure. Default is 300.
        show_legend : bool, optional
            Whether to show the legend. Default is False.
        axis : matplotlib.pyplot.axis, optional
            The axis to use for plotting the data. If None, a new axis is created.
        xlabel : str, optional
            Label for the x-axis. If not provided, it defaults to the x parameter.
        ylabel : str, optional
            Label for the y-axis. If not provided, it defaults to the y parameter.
        vertical_line_color : str, optional
            Color of the vertical line. Default is 'k' (black).
        vertical_line_width : float, optional
            Width of the vertical line. Default is 0.5.
        marker_size : float, optional
            Size of the markers. Default is 0.8.
        clabel : str, optional
            Label for the color bar. If not provided, it defaults to color_column.
        legend_font_size : float, optional
            Font size for the legend. Default is 6.
        no_xticks : bool, optional
            If True, x-axis ticks are not shown. Default is False.
        no_yticks : bool, optional
            If True, y-axis ticks are not shown. Default is False.
        no_cbar : bool, optional
            If True, the color bar is not shown. Default is False.
        no_xlabel : bool, optional
            If True, the x-axis label is not shown. Default is False.
        no_ylabel : bool, optional
            If True, the y-axis label is not shown. Default is False.
        relative_color_values : bool, optional
            If True, color values are shown relative to their minimum value. Default is False.
        dataframe : pandas.DataFrame, optional
            Dataframe containing the data. If not provided, self.rosetta_docking is used.
        separator : str, optional
            Separator used in the plot label. Default is '_'.
        **kwargs : additional keyword arguments
            Additional arguments to pass to the scatter function.

        Returns
        =======
        axis : matplotlib.pyplot.axis
            The axis object of the plot, if return_axis is True.

        Raises
        ======
        ValueError
            If the specified model or ligand is not found in the data.
        """

        def _addDistanceAndAngleData(ligand_series, model, ligand, dataframe):
            if model in self.rosetta_docking_distances:
                if ligand in self.rosetta_docking_distances[model]:
                    if self.rosetta_docking_distances[model][ligand] is not None:
                        for distance in self.rosetta_docking_distances[model][ligand]:
                            if dataframe is not None:
                                index_columns = ['Model', 'Ligand', 'Pose']
                                indexes = dataframe.reset_index().set_index(index_columns).index
                                ligand_series[distance] = self.rosetta_docking_distances[model][ligand][self.rosetta_docking_distances[model][ligand].index.isin(indexes)][distance].tolist()
                            else:
                                ligand_series[distance] = self.rosetta_docking_distances[model][ligand][distance].tolist()

            if model in self.rosetta_docking_angles:
                if ligand in self.rosetta_docking_angles[model]:
                    if self.rosetta_docking_angles[model][ligand] is not None:
                        for angle in self.rosetta_docking_angles[model][ligand]:
                            if dataframe is not None:
                                index_columns = ['Model', 'Ligand', 'Pose']
                                indexes = dataframe.reset_index().set_index(index_columns).index
                                ligand_series[angle] = self.rosetta_docking_angles[model][ligand][self.rosetta_docking_angles[model][ligand].index.isin(indexes)][angle].tolist()
                            else:
                                ligand_series[angle] = self.rosetta_docking_angles[model][ligand][angle].tolist()

            return ligand_series

        def _filterByMetrics(ligand_series, metrics):
            for metric, value in metrics.items():
                if isinstance(value, float):
                    mask = ligand_series[metric] <= value
                elif isinstance(value, tuple):
                    mask = (ligand_series[metric] >= value[0]) & (ligand_series[metric] <= value[1])
                ligand_series = ligand_series[mask]
            return ligand_series

        def _filterByLabels(ligand_series, labels):
            for label, value in labels.items():
                if value is not None:
                    mask = ligand_series[label] == value
                    ligand_series = ligand_series[mask]
            return ligand_series

        def _defineColorColumns(ligand_series):
            color_columns = [col for col in ligand_series.columns if ':' not in col and 'distance' not in col and 'angle' not in col and not col.startswith('metric_')]
            return color_columns

        def _plotScatter(axis, ligand_series, x, y, color_column, color_columns, plot_label, clim, marker_size, size, **kwargs):
            if color_column is not None:
                if clim is not None:
                    vmin, vmax = clim
                else:
                    vmin, vmax = None, None

                ascending = False
                colormap = 'Blues_r'

                if color_column == 'Step':
                    ascending = True
                    colormap = 'Blues'

                elif color_column in ['Epoch', 'Cluster']:
                    ascending = True
                    color_values = ligand_series.reset_index()[color_column]
                    cmap = plt.cm.jet
                    cmaplist = [cmap(i) for i in range(cmap.N)]
                    cmaplist[0] = (.5, .5, .5, 1.0)
                    max_value = max(color_values.tolist())
                    bounds = np.linspace(0, max_value + 1, max_value + 2)
                    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                    colormap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
                    color_values = color_values + 0.01
                    sc = axis.scatter(ligand_series[x], ligand_series[y], c=color_values, cmap=colormap, norm=norm, vmin=vmin, vmax=vmax, label=plot_label, s=marker_size*size, **kwargs)
                    if not no_cbar:
                        cbar = plt.colorbar(sc, ax=axis)
                        cbar.set_label(label=color_column, size=labels_size * size)
                        cbar.ax.tick_params(labelsize=labels_size * size)

                elif color_column in color_columns:
                    ligand_series = ligand_series.sort_values(color_column, ascending=ascending)
                    color_values = ligand_series[color_column]

                    if relative_color_values:
                        color_values = color_values - np.min(color_values)

                    sc = axis.scatter(ligand_series[x], ligand_series[y], c=color_values, cmap=colormap, vmin=vmin, vmax=vmax, label=plot_label, s=marker_size * size, **kwargs)
                    if not no_cbar:
                        cbar = plt.colorbar(sc, ax=axis)
                        if clabel is None:
                            clabel = color_column
                        cbar.set_label(label=clabel, size=labels_size * size)
                        cbar.ax.tick_params(labelsize=labels_size * size)
                else:
                    sc = axis.scatter(ligand_series[x], ligand_series[y], c=color_column, vmin=vmin, vmax=vmax, label=plot_label, s=marker_size * size, **kwargs)
            else:
                sc = axis.scatter(ligand_series[x], ligand_series[y], label=plot_label, s=marker_size * size, **kwargs)
            return sc

        # Extract model series from dataframe or self.rosetta_docking
        if dataframe is not None:
            model_series = dataframe[dataframe.index.get_level_values('Model') == model]
        else:
            model_series = self.rosetta_docking[self.rosetta_docking.index.get_level_values('Model') == model]

        if model_series.empty:
            raise ValueError(f'Model name {model} not found in data!')

        ligand_series = model_series[model_series.index.get_level_values('Ligand') == ligand]
        if ligand_series.empty:
            raise ValueError(f"Ligand name {ligand} not found in model's {model} data!")

        # Add distance and angle data to ligand_series
        if len(ligand_series) != 0:
            ligand_series = _addDistanceAndAngleData(ligand_series, model, ligand, dataframe)

        # Filter points by metrics
        if metrics is not None:
            ligand_series = _filterByMetrics(ligand_series, metrics)

        # Filter points by labels
        if labels is not None:
            ligand_series = _filterByLabels(ligand_series, labels)

        # Check if an axis has been given
        new_axis = False
        if axis is None:
            plt.figure(figsize=(4*size, 3.3*size), dpi=dpi)
            axis = plt.gca()
            new_axis = True

        # Define plot label
        if plot_label is None:
            plot_label = f"{model}{separator}{ligand}"

        # Define color columns
        color_columns = _defineColorColumns(ligand_series)

        # Plot scatter
        sc = _plotScatter(axis, ligand_series, x, y, color_column, color_columns, plot_label, clim, marker_size, size, **kwargs)

        # Plot vertical line if specified
        if vertical_line is not None:
            axis.axvline(vertical_line, c=vertical_line_color, lw=vertical_line_width, ls='--')

        # Set labels and title
        if xlabel is None and not no_xlabel:
            xlabel = x
        if ylabel is None and not no_ylabel:
            ylabel = y

        axis.set_xlabel(xlabel, fontsize=labels_size*size)
        axis.set_ylabel(ylabel, fontsize=labels_size*size)
        axis.tick_params(axis='both', labelsize=labels_size*size)

        # Set ticks visibility
        if no_xticks:
            axis.set_xticks([])
        if no_yticks:
            axis.set_yticks([])

        if title is not None:
            axis.set_title(title, fontsize=title_size*size)
        if xlim is not None:
            axis.set_xlim(xlim)
        if ylim is not None:
            axis.set_ylim(ylim)

        if show_legend:
            axis.legend(prop={'size': legend_font_size*size})

        if show:
            plt.show()

        if return_axis:
            return axis

    def rosettaDockingCatalyticBindingFreeEnergyMatrix(self, initial_threshold=3.5, initial_threshold_filter=3.5, measured_metrics=None,
                                                       store_values=False, lig_label_rot=90, observable='interface_delta_B',
                                                       matrix_file='catalytic_matrix.npy', models_file='catalytic_models.json',
                                                       max_metric_threshold=30, pele_data=None, KT=5.93, to_csv=None,
                                                       only_proteins=None, only_ligands=None, average_binding_energy=False,
                                                       nan_to_zero=False):

        def _bindingFreeEnergyMatrix(KT=KT, sort_by_ligand=None, models_file='catalytic_models.json',
                                     lig_label_rot=90, pele_data=None, only_proteins=None, only_ligands=None,
                                     abc=False, avg_ebc=False, n_poses=10, **metrics):

            metrics_filter = {m: metrics[m] for m in metrics if m.startswith('metric_')}
            labels_filter = {l: metrics[l] for l in metrics if l.startswith('label_')}

            if pele_data is None:
                pele_data = self.rosetta_docking

            if only_proteins is not None:
                proteins = [p for p in pele_data.index.get_level_values('Model').unique() if p in only_proteins]
            else:
                proteins = pele_data.index.get_level_values('Model').unique()

            if only_ligands is not None:
                ligands = [l for l in pele_data.index.get_level_values('Ligand').unique() if l in only_ligands]
            else:
                ligands = pele_data.index.get_level_values('Ligand').unique()

            if len(proteins) == 0:
                raise ValueError('No proteins were found!')
            if len(ligands) == 0:
                raise ValueError('No ligands were found!')

            # Create a matrix of length proteins times ligands
            M = np.zeros((len(proteins), len(ligands)))

            # Calculate the probability of each state
            for i, protein in enumerate(proteins):
                protein_series = pele_data[pele_data.index.get_level_values('Model') == protein]

                for j, ligand in enumerate(ligands):
                    ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]

                    if not ligand_series.empty:

                        if abc:
                            # Calculate partition function
                            total_energy = ligand_series['total_score']
                            energy_minimum = total_energy.min()
                            relative_energy = total_energy - energy_minimum
                            Z = np.sum(np.exp(-relative_energy / KT))

                        # Calculate catalytic binding energy
                        catalytic_series = ligand_series

                        for metric in metrics_filter:
                            if isinstance(metrics_filter[metric], float):
                                mask = catalytic_series[metric] <= metrics_filter[metric]
                            elif isinstance(metrics_filter[metric], tuple):
                                mask = (catalytic_series[metric] >= metrics_filter[metric][0]).to_numpy()
                                mask = mask & ((catalytic_series[metric] <= metrics_filter[metric][1]).to_numpy())
                            catalytic_series = catalytic_series[mask]

                        for l in labels_filter:
                            # Filter by labels
                            if labels_filter[l] is not None:
                                catalytic_series = catalytic_series[catalytic_series[l] == labels_filter[l]]

                        if abc:
                            total_energy = catalytic_series['total_score']
                            relative_energy = total_energy - energy_minimum
                            probability = np.exp(-relative_energy / KT) / Z
                            M[i][j] = np.sum(probability * catalytic_series[observable])
                        elif avg_ebc:
                            M[i][j] = catalytic_series.nsmallest(n_poses, observable)[observable].mean()
                    else:
                        M[i][j] = np.nan

            if nan_to_zero:
                M[np.isnan(M)] = 0.0

            if abc:
                binding_metric_label = '$A_{B}^{C}$'
            elif avg_ebc:
                binding_metric_label = '$\overline{E}_{B}^{C}$'
            else:
                raise ValueError('You should mark at least one option: $A_{B}^{C}$ or $\overline{E}_{B}^{C}$!')

            if store_values:
                np.save(matrix_file, M)
                if not models_file.endswith('.json'):
                    models_file = models_file + '.json'
                with open(models_file, 'w') as of:
                    json.dump(list(proteins), of)

            if to_csv is not None:
                catalytic_values = {
                    'Model': [],
                    'Ligand': [],
                    binding_metric_label: []
                }

                for i, m in zip(M, proteins):
                    for v, l in zip(i, ligands):
                        catalytic_values['Model'].append(m)
                        catalytic_values['Ligand'].append(l)
                        catalytic_values[binding_metric_label].append(v)
                catalytic_values = pd.DataFrame(catalytic_values)
                catalytic_values.set_index(['Model', 'Ligand'])
                catalytic_values.to_csv(to_csv)

            # Sort matrix by ligand or protein
            if sort_by_ligand == 'by_protein':
                protein_labels = proteins
            else:
                ligand_index = list(ligands).index(sort_by_ligand)
                sort_indexes = M[:, ligand_index].argsort()
                M = M[sort_indexes]
                protein_labels = [proteins[x] for x in sort_indexes]

            plt.figure(dpi=100, figsize=(0.28 * len(ligands), 0.2 * len(proteins)))
            plt.imshow(M, cmap='autumn')
            plt.colorbar(label=binding_metric_label)

            plt.xlabel('Ligands', fontsize=12)
            ax = plt.gca()
            ax.set_xticks(np.arange(len(ligands)))  # Set tick positions
            ax.set_xticklabels(ligands, rotation=lig_label_rot)
            plt.xticks(np.arange(len(ligands)), ligands, rotation=lig_label_rot)
            plt.ylabel('Proteins', fontsize=12)
            ax.set_yticks(np.arange(len(proteins)))  # Set tick positions
            plt.yticks(np.arange(len(proteins)), protein_labels)

            display(plt.show())

        # Check to_csv input
        if to_csv is not None and not isinstance(to_csv, str):
            raise ValueError('to_csv must be a path to the output csv file.')
        if to_csv is not None and not to_csv.endswith('.csv'):
            to_csv = to_csv + '.csv'

        # Define if PELE data is given
        if pele_data is None:
            pele_data = self.rosetta_docking

        # Add checks for the given pele data pandas df
        metrics = [k for k in pele_data.keys() if 'metric_' in k]
        labels = {}
        for m in metrics:
            for l in pele_data.keys():
                if 'label_' in l and l.replace('label_', '') == m.replace('metric_', ''):
                    labels[m] = sorted(list(set(pele_data[l])))

        metrics_sliders = {}
        labels_ddms = {}
        for m in metrics:
            if measured_metrics is not None:
                if m in measured_metrics:
                    threshold = initial_threshold
                else:
                    threshold = initial_threshold_filter
            else:
                threshold = initial_threshold_filter  # Ensure threshold is always defined

            if self.rosetta_docking_metric_type[m] == 'distance':
                m_slider = FloatSlider(
                    value=threshold,
                    min=0,
                    max=max_metric_threshold,
                    step=0.1,
                    description=m + ':',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.2f',
                    style={'description_width': 'initial'})

            elif self.rosetta_docking_metric_type[m] == 'angle':
                m_slider = FloatRangeSlider(
                    value=[110, 130],
                    min=-180,
                    max=180,
                    step=0.1,
                    description=m + ':',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.2f',
                )

            metrics_sliders[m] = m_slider

            if m in labels and labels[m] != []:
                label_options = [None] + labels[m]
                label_ddm = Dropdown(options=label_options, description=m.replace('metric_', 'label_'), style={'description_width': 'initial'})
                metrics_sliders[m.replace('metric_', 'label_')] = label_ddm

        if only_proteins is not None:
            if isinstance(only_proteins, str):
                only_proteins = [only_proteins]

        # Get only ligands if given
        if only_ligands is not None:
            if isinstance(only_ligands, str):
                only_ligands = [only_ligands]

            ligands = [l for l in self.rosetta_docking.index.get_level_values('Ligand').unique() if l in only_ligands]
        else:
            ligands = self.rosetta_docking.index.get_level_values('Ligand').unique()

        VB = []
        ligand_ddm = Dropdown(options=list(ligands) + ['by_protein'], description='Sort by ligand',
                              style={'description_width': 'initial'})
        VB.append(ligand_ddm)

        abc = Checkbox(value=True, description='$A_{B}^{C}$')
        VB.append(abc)

        if average_binding_energy:
            avg_ebc = Checkbox(value=False, description='$\overline{E}_{B}^{C}$')
            VB.append(avg_ebc)

            Ebc_slider = IntSlider(
                value=10,
                min=1,
                max=1000,
                step=1,
                description='N poses (only $\overline{E}_{B}^{C}$):',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True)
            VB.append(Ebc_slider)

        KT_slider = FloatSlider(
            value=KT,
            min=0.593,
            max=1000.0,
            step=0.1,
            description='KT:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f')

        for m in metrics_sliders:
            VB.append(metrics_sliders[m])
        for m in labels_ddms:
            VB.append(labels_ddms[m])
        VB.append(KT_slider)

        if average_binding_energy:
            plot = interactive_output(_bindingFreeEnergyMatrix, {'KT': KT_slider, 'sort_by_ligand': ligand_ddm,
                                      'pele_data': fixed(pele_data), 'models_file': fixed(models_file),
                                      'lig_label_rot': fixed(lig_label_rot), 'only_proteins': fixed(only_proteins),
                                      'only_ligands': fixed(only_ligands), 'abc': abc, 'avg_ebc': avg_ebc,
                                      'n_poses': Ebc_slider, **metrics_sliders})
        else:
            plot = interactive_output(_bindingFreeEnergyMatrix, {'KT': KT_slider, 'sort_by_ligand': ligand_ddm,
                                      'pele_data': fixed(pele_data), 'models_file': fixed(models_file),
                                      'lig_label_rot': fixed(lig_label_rot), 'only_proteins': fixed(only_proteins),
                                      'only_ligands': fixed(only_ligands), 'abc': abc, **metrics_sliders})

        VB.append(plot)
        VB = VBox(VB)

        display(VB)

    def computeRosettaDockingFreeEnergy(self, df=None, KT: float = 0.593,
                                        binding_col: Optional[str] = None,
                                        add_probabilities: bool = False,
                                        skip_warnings: bool = False) -> pd.DataFrame:
        """
        Compute Boltzmann-weighted binding free energy per (Model, Ligand) from docking poses.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Rosetta docking dataframe indexed by (Model, Ligand, Pose). If None, uses
            ``self.rosetta_docking_data``.
        KT : float, optional
            Thermal energy in kcal/mol. Default is 0.593 (298 K).
        binding_col : str, optional
            Column containing per-pose binding energies. If None, the first column whose
            name starts with ``'interface_delta_'`` is used.
        add_probabilities : bool, optional
            If True, store Boltzmann probabilities per pose in a ``boltzmann_p`` column
            on the provided dataframe (or ``self.rosetta_docking_data`` when df is None).
        skip_warnings : bool, optional
            Suppress non-fatal warnings (auto-selected binding column, NaNs, empty conformers) when True.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by (Model, Ligand) with a single column ``free_energy``.
        """
        target_df = df if df is not None else getattr(self, "rosetta_docking_data", None)
        if target_df is None:
            raise ValueError("No docking dataframe provided and self.rosetta_docking_data is missing.")

        if not isinstance(target_df.index, pd.MultiIndex) or len(target_df.index.names) != 3:
            raise ValueError("Input must be a MultiIndex dataframe with levels (Model, Ligand, Pose).")

        if KT <= 0:
            raise ValueError("KT must be positive.")

            if binding_col is None:
                binding_col = next((c for c in target_df.columns if c.startswith("interface_delta_")), None)
                if binding_col is None:
                    raise ValueError("Could not find a binding energy column starting with 'interface_delta_'.")
                binding_col_display = binding_col if isinstance(binding_col, str) else repr(binding_col)
                if not skip_warnings:
                    warnings.warn(f"Using binding energy column '{binding_col_display}'.", UserWarning)

        if binding_col not in target_df.columns:
            raise ValueError(f"Binding energy column '{binding_col}' not found in dataframe.")

        # Mutate the caller only when probabilities are requested; otherwise work on a copy
        df_for_probs = target_df if add_probabilities else target_df.copy()

        energies = pd.to_numeric(df_for_probs[binding_col], errors='coerce')
        if energies.isna().any() and not skip_warnings:
            warnings.warn("Binding energies contain NaNs; affected poses will be ignored in free energy calculation.", UserWarning)

        free_energy_records = []
        if add_probabilities and "boltzmann_p" not in df_for_probs.columns:
            df_for_probs["boltzmann_p"] = np.nan

        for (model, ligand), group in df_for_probs.groupby(level=[0, 1]):
            e = pd.to_numeric(group[binding_col], errors='coerce')
            finite_mask = np.isfinite(e)
            if not finite_mask.any():
                free_energy_records.append(((model, ligand), np.nan))
                continue

            valid_e = e[finite_mask].to_numpy(dtype=float)
            e0 = valid_e.min()
            shifted = valid_e - e0
            weights = np.exp(-shifted / KT)
            Z = weights.sum()
            free_energy = e0 - KT * np.log(Z)
            free_energy_records.append(((model, ligand), free_energy))

            if add_probabilities:
                probs = weights / Z
                prob_series = pd.Series(probs, index=e[finite_mask].index)
                df_for_probs.loc[prob_series.index, "boltzmann_p"] = prob_series

        free_energy_df = pd.DataFrame.from_records(
            [(idx[0], idx[1], val) for idx, val in free_energy_records],
            columns=["Model", "Ligand", "free_energy"],
        ).set_index(["Model", "Ligand"])

        # If we updated self.rosetta_docking_data in-place, keep the changes
        if add_probabilities and df is None:
            self.rosetta_docking_data = df_for_probs

        return free_energy_df

    def getBestRosettaDockingPoses(
            self,
            filter_values,
            return_failed=False,
            exclude_models=None,
            exclude_ligands=None,
            exclude_pairs=None,
            score_column='interface_delta_B',
            docking_df=None,
        ):
        """
        Get best docking poses based on the best SCORE and a set of metrics with specified thresholds.
        The filter thresholds must be provided with a dictionary using the metric names as keys
        and the thresholds as the values.

        Parameters
        ==========
        filter_values : dict
            Thresholds for the filter.
        return_failed : bool
            Whether to return a list of the dockings without any models fulfilling
            the selection criteria. It is returned as a tuple (index 0) alongside
            the filtered data frame (index 1).
        exclude_models : list, optional
            List of models to be excluded from the selection.
        exclude_ligands : list, optional
            List of ligands to be excluded from the selection.
        exclude_pairs : list, optional
            List of pair tuples (model, ligand) to be excluded from the selection.
        score_column : str, optional
            Column name to use for scoring. Default is 'interface_delta_B'.

        Parameters
        ==========
        docking_df : pandas.DataFrame, optional
            Custom Rosetta docking results to analyze. Must be indexed by (Model, Ligand, Pose)
            and contain the requested metric columns. Defaults to ``self.rosetta_docking``.

        Returns
        =======
        pandas.DataFrame
            Dataframe containing the best poses based on the given criteria.
        """

        if exclude_models is None:
            exclude_models = []
        if exclude_ligands is None:
            exclude_ligands = []
        if exclude_pairs is None:
            exclude_pairs = []

        source_df = docking_df if docking_df is not None else self.rosetta_docking
        if source_df is None or source_df.empty:
            raise ValueError("No Rosetta docking data available. Run analyseRosettaDocking first.")

        best_poses = []
        failed = []

        for model in source_df.index.get_level_values('Model').unique():

            if model in exclude_models:
                continue

            model_series = source_df[source_df.index.get_level_values("Model") == model]

            for ligand in model_series.index.get_level_values('Ligand').unique():

                if ligand in exclude_ligands:
                    continue

                if (model, ligand) in exclude_pairs:
                    continue

                ligand_data = model_series[
                    model_series.index.get_level_values("Ligand") == ligand
                ]

                for metric, threshold in filter_values.items():

                    if metric not in [score_column, "RMSD"]:
                        if not metric.startswith("metric_"):
                            metric_label = "metric_" + metric
                        else:
                            metric_label = metric

                        if isinstance(threshold, (float, int)):
                            ligand_data = ligand_data[ligand_data[metric_label] <= threshold]
                        elif isinstance(threshold, (tuple, list)):
                            ligand_data = ligand_data[
                                (ligand_data[metric_label] >= threshold[0]) &
                                (ligand_data[metric_label] <= threshold[1])
                            ]
                    else:
                        metric_label = metric
                        ligand_data = ligand_data[ligand_data[metric_label] < threshold]

                if ligand_data.empty:
                    failed.append((model, ligand))
                    continue

                best_pose_idx = ligand_data[score_column].idxmin()
                best_poses.append(best_pose_idx)

        best_poses_df = source_df.loc[best_poses]

        if return_failed:
            return failed, best_poses_df

        return best_poses_df

    def getBestRosettaDockingPosesIteratively(
        self,
        metrics,
        ligands=None,
        distance_step=0.1,
        angle_step=1.0,
        fixed=None,
        max_distance=None,
        max_distance_step_shift=None,
        verbose=False,
        score_column='interface_delta_B',
        docking_df=None,
    ):
        """
        Iteratively select the best Rosetta docking poses for each model/ligand pair.

        This mirrors :meth:`getBestDockingPosesIteratively`, but operates on
        ``self.rosetta_docking`` (or a user-supplied DataFrame) and uses Rosetta metrics. Thresholds are gradually
        relaxed (except for metrics listed in ``fixed``) until at least one pose is
        accepted per model/ligand pair or no further relaxation is possible.

        Parameters
        ----------
        docking_df : pandas.DataFrame, optional
            Custom Rosetta docking results to analyze. Must be indexed by (Model, Ligand, Pose)
            and contain the requested metric columns. Defaults to ``self.rosetta_docking``.
        """

        source_df = docking_df if docking_df is not None else self.rosetta_docking

        if source_df is None or source_df.empty:
            raise ValueError("No Rosetta docking data available. Run analyseRosettaDocking first.")

        if fixed is None:
            fixed = []
        elif isinstance(fixed, str):
            fixed = [fixed]

        metrics = metrics.copy()
        non_fixed_metrics = set(metrics.keys()) - set(fixed)
        if not non_fixed_metrics:
            raise ValueError("You must leave at least one metric not fixed")

        if ligands is not None:
            data = source_df[source_df.index.get_level_values(1).isin(ligands)]
        else:
            data = source_df

        protein_ligand_pairs = set(zip(
            data.index.get_level_values(0),
            data.index.get_level_values(1)
        ))

        # Ensure metric types are known for every requested metric
        for metric in metrics.keys():
            metric_label = metric if metric.startswith('metric_') else 'metric_' + metric
            metric_type = self._get_rosetta_metric_type(metric_label)
            if metric_type is None:
                inferred_type = None
                if metric_label in self.rosetta_docking.columns:
                    if 'angle' in metric_label.lower() or 'torsion' in metric_label.lower():
                        inferred_type = 'angle'
                    else:
                        inferred_type = 'distance'
                if inferred_type is not None:
                    self._set_rosetta_metric_type(metric_label, inferred_type)
                else:
                    raise ValueError(
                        f"Metric type for {metric_label} not defined. "
                        "Ensure metrics were created via combineRosettaDockingDistancesIntoMetrics "
                        "or populate self.rosetta_docking_metric_type manually."
                    )

        extracted_pairs = set()
        selected_indexes = []
        current_distance_step = distance_step
        step_shift_applied = False

        while len(extracted_pairs) < len(protein_ligand_pairs):
            if verbose:
                ti = time.time()

            best_poses = self.getBestRosettaDockingPoses(metrics, score_column=score_column, docking_df=data)

            new_selected_pairs = set()
            for idx in best_poses.index:
                pair = (idx[0], idx[1])
                if pair not in extracted_pairs:
                    selected_indexes.append(idx)
                    new_selected_pairs.add(pair)

            extracted_pairs.update(new_selected_pairs)

            if len(extracted_pairs) >= len(protein_ligand_pairs):
                break

            remaining_pairs = protein_ligand_pairs - extracted_pairs
            mask = [((idx[0], idx[1]) in remaining_pairs) for idx in data.index]
            remaining_data = data[mask]

            if remaining_data.empty:
                break

            metric_acceptance = {}
            for metric in metrics:
                if metric in fixed:
                    continue
                metric_label = metric if metric.startswith('metric_') else 'metric_' + metric
                metric_type = self._get_rosetta_metric_type(metric_label)
                if metric_type is None:
                    raise ValueError(f"Metric type for {metric_label} not defined.")

                metric_values = remaining_data[metric_label]

                threshold = metrics[metric]
                if isinstance(threshold, (int, float)):
                    if metric_type in ['distance', 'angle']:
                        acceptance = metric_values <= threshold
                    else:
                        acceptance = metric_values >= threshold
                elif isinstance(threshold, (tuple, list)):
                    lower, upper = threshold
                    acceptance = (metric_values >= lower) & (metric_values <= upper)
                else:
                    raise ValueError(f"Invalid threshold type for metric {metric}")

                metric_acceptance[metric] = acceptance.sum()

            ordered_metrics = sorted(
                [(m, a) for m, a in metric_acceptance.items() if m not in fixed],
                key=lambda x: x[1]
            )

            updated = False
            for metric, _ in ordered_metrics:
                metric_label = metric if metric.startswith('metric_') else 'metric_' + metric
                metric_type = self._get_rosetta_metric_type(metric_label)
                if metric_type == 'distance':
                    step = current_distance_step
                elif metric_type == 'angle':
                    step = angle_step
                else:
                    raise ValueError(f"Unknown metric type for {metric_label}")

                threshold = metrics[metric]
                if isinstance(threshold, (int, float)):
                    new_value = threshold + step

                    if metric_type == 'distance' and max_distance is not None:
                        if not step_shift_applied and new_value >= max_distance:
                            if max_distance_step_shift is not None:
                                current_distance_step = max_distance_step_shift
                                step_shift_applied = True
                                if verbose:
                                    print(
                                        f"Max distance {max_distance} reached for metric {metric}. "
                                        f"Applying step shift to {current_distance_step}."
                                    )
                            else:
                                new_value = max_distance
                                metrics[metric] = new_value
                                if verbose:
                                    print(
                                        f"Max distance {max_distance} reached for metric {metric}. "
                                        "Terminating iteration."
                                    )
                                updated = True
                                break

                    metrics[metric] = new_value
                    updated = True
                    break

                elif isinstance(threshold, (tuple, list)):
                    lower, upper = threshold
                    metrics[metric] = (lower - step, upper + step)
                    updated = True
                    break
                else:
                    raise ValueError(f"Invalid threshold type for metric {metric}")

            if not updated:
                if verbose:
                    print("No metrics were updated. Terminating iteration.")
                break

            if verbose:
                elapsed_time = time.time() - ti
                print(
                    f"Selected pairs: {len(extracted_pairs)}/{len(protein_ligand_pairs)}, "
                    f"Current thresholds: {metrics}, Time elapsed: {elapsed_time:.2f}s",
                    end='\r'
                )

        if selected_indexes:
            best_poses = data.loc[selected_indexes]
        else:
            best_poses = pd.DataFrame()

        return best_poses

    def extractRosettaDockingModels(self, docking_folder, input_df, output_folder, separator='_'):
        """
        Extract models based on an input DataFrame with index ['Model', 'Ligand', 'Pose'].

        Parameters
        ==========
        docking_folder : str
            Path to folder where the Rosetta docking files are contained.
        input_df : pd.DataFrame
            DataFrame containing the models to be extracted with index ['Model', 'Ligand', 'Pose'].
        separator : str
            Separator character used in file names. Default is '-'.

        Returns
        =======
        list
            List of models extracted.
        """

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        executable = "extract_pdbs.linuxgccrelease"
        models = []
        missing_models = []

        shared_params = []
        params_folder = os.path.join(docking_folder, "params")
        if os.path.isdir(params_folder):
            for entry in sorted(os.listdir(params_folder)):
                if entry.endswith(".params"):
                    shared_params.append(os.path.join(params_folder, entry))

        ligand_params = {}
        if self.rosetta_docking is not None:
            ligands = self.rosetta_docking.index.levels[1]
        else:
            ligands = []
        for ligand in ligands:
            ligand_folder = os.path.join(docking_folder, "ligand_params", ligand)
            if os.path.exists(ligand_folder):
                ligand_params[ligand] = os.path.join(ligand_folder, f"{ligand}.params")

        pose_padding_cache: Dict[Tuple[str, str], int] = {}

        def _infer_pose_padding(path: str, model_name: str) -> int:
            """
            Determine zero-padding by reading SCORE tags for this model.
            """
            padding = 4
            try:
                with open(path) as sf:
                    for line in sf:
                        if not line.startswith("SCORE:"):
                            continue
                        parts = line.strip().split()
                        if not parts or parts[-1] == "description":
                            continue
                        tag = parts[-1]
                        if not tag.startswith(f"{model_name}_"):
                            continue
                        suffix = tag[len(model_name) + 1 :]
                        if suffix.isdigit():
                            padding = max(padding, len(suffix))
            except OSError:
                return padding
            return padding

        for index, row in input_df.iterrows():
            model, ligand, pose = index

            output_models_root = os.path.join(docking_folder, "output_models")
            sep_candidates = []
            if separator:
                sep_candidates.append(separator)
            sep_candidates.extend(['-', '_'])
            seen = set()
            sep_candidates = [s for s in sep_candidates if not (s in seen or seen.add(s))]

            output_model_dir = None
            used_separator = None
            for sep_candidate in sep_candidates:
                candidate_dir = os.path.join(output_models_root, f"{model}{sep_candidate}{ligand}")
                if os.path.exists(candidate_dir):
                    output_model_dir = candidate_dir
                    used_separator = sep_candidate
                    break

            if output_model_dir is None:
                missing_models.append(f"{model}{separator}{ligand}")
                continue

            silent_file = os.path.join(output_model_dir, f"{model}{used_separator}{ligand}.out")
            if not os.path.exists(silent_file):
                missing_models.append(f"{model}{used_separator}{ligand}")
                continue

            if 'description' in row.index:
                best_model_tag = row['description']
            else:
                padding_key = (silent_file, model)
                pose_padding = pose_padding_cache.get(padding_key)
                if pose_padding is None:
                    pose_padding = _infer_pose_padding(silent_file, model)
                    pose_padding_cache[padding_key] = pose_padding

                if isinstance(pose, (int, float)) and not isinstance(pose, bool):
                    pose_value = int(pose)
                    pose_str = str(pose_value).zfill(pose_padding)
                else:
                    pose_str = str(pose)
                best_model_tag = f"{model}_{pose_str}"

            command = f"{executable} -silent {silent_file} -tags {best_model_tag}"
            for param_file in shared_params:
                command += f" -extra_res_fa {param_file} "
            if ligand in ligand_params:
                command += f" -extra_res_fa {ligand_params[ligand]} "
            command += " 2>/dev/null"
            exit_code = os.system(command)

            pdb_filename = f"{best_model_tag}.pdb"

            if exit_code == 0 and os.path.exists(pdb_filename):
                shutil.move(pdb_filename, os.path.join(output_folder, pdb_filename))
                models.append(os.path.join(output_folder, pdb_filename))
            else:
                print(f"Failed to extract pose '{best_model_tag}' from {silent_file}.")

        self.getModelsSequences()

        if missing_models:
            print("Missing models in Rosetta Docking folder:")
            print("\t" + ", ".join(missing_models))

        return models

    def extractRosettaModels(self, rosetta_folder, input_df, output_folder, overwrite=False):
        """
        Extract Rosetta models (e.g., relax runs) using an input DataFrame indexed by
        ['Model', 'Pose'] as returned by :meth:`analyseRosettaCalculation`.

        Parameters
        ----------
        rosetta_folder : str
            Path to the Rosetta calculation folder containing ``output_models/<model>/*.out`` files.
        input_df : pd.DataFrame
            DataFrame listing the poses to extract. Its index must contain ``Model`` and ``Pose``.
            Rows including a ``description`` column will use that tag directly.
        output_folder : str
            Destination directory where the extracted PDBs will be copied.
        overwrite : bool, optional
            If True, overwrite existing files in ``output_folder``. Otherwise, skip already extracted poses.

        Returns
        -------
        list
            Paths to the extracted PDB files.
        """

        rosetta_folder = os.path.abspath(rosetta_folder)
        output_folder = os.path.abspath(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        executable = "extract_pdbs.linuxgccrelease"
        models = []
        missing_models = []

        extra_res_files = []
        params_dir = os.path.join(rosetta_folder, "params")
        if os.path.isdir(params_dir):
            for entry in sorted(os.listdir(params_dir)):
                if entry.endswith(".params"):
                    extra_res_files.append(os.path.join(params_dir, entry))

        ligand_params_root = os.path.join(rosetta_folder, "ligand_params")
        if os.path.isdir(ligand_params_root):
            for ligand in sorted(os.listdir(ligand_params_root)):
                ligand_dir = os.path.join(ligand_params_root, ligand)
                if not os.path.isdir(ligand_dir):
                    continue
                for entry in sorted(os.listdir(ligand_dir)):
                    if entry.endswith(".params"):
                        extra_res_files.append(os.path.join(ligand_dir, entry))

        pose_padding_cache: Dict[Tuple[str, str], int] = {}

        def _infer_pose_padding(path: str, model_name: str) -> int:
            padding = 4
            try:
                with open(path) as sf:
                    for line in sf:
                        if not line.startswith("SCORE:"):
                            continue
                        parts = line.strip().split()
                        if not parts or parts[-1] == "description":
                            continue
                        tag = parts[-1]
                        if not tag.startswith(f"{model_name}_"):
                            continue
                        suffix = tag[len(model_name) + 1 :]
                        if suffix.isdigit():
                            padding = max(padding, len(suffix))
            except OSError:
                return padding
            return padding

        extraction_jobs: Dict[str, List[str]] = {}

        for index, row in input_df.iterrows():
            if isinstance(index, tuple):
                model = index[0]
                pose = index[1]
            else:
                raise ValueError("Input dataframe index must contain (Model, Pose).")

            model_dir = os.path.join(rosetta_folder, "output_models", str(model))
            if not os.path.isdir(model_dir):
                missing_models.append(str(model))
                continue

            preferred_silent = os.path.join(model_dir, f"{model}_relax.out")
            silent_file = preferred_silent if os.path.exists(preferred_silent) else None
            if silent_file is None:
                for entry in sorted(os.listdir(model_dir)):
                    if entry.endswith(".out"):
                        silent_file = os.path.join(model_dir, entry)
                        break

            if silent_file is None:
                missing_models.append(str(model))
                continue

            if "description" in row.index:
                best_model_tag = row["description"]
            else:
                padding_key = (silent_file, str(model))
                pose_padding = pose_padding_cache.get(padding_key)
                if pose_padding is None:
                    pose_padding = _infer_pose_padding(silent_file, str(model))
                    pose_padding_cache[padding_key] = pose_padding

                if isinstance(pose, (int, float)) and not isinstance(pose, bool):
                    pose_value = int(pose)
                    pose_str = str(pose_value).zfill(pose_padding)
                else:
                    pose_str = str(pose)
                best_model_tag = f"{model}_{pose_str}"

            extraction_jobs.setdefault(silent_file, []).append(best_model_tag)

        for silent_file, tags in extraction_jobs.items():
            if overwrite:
                tags_to_extract = tags
            else:
                tags_to_extract = [
                    tag
                    for tag in tags
                    if not os.path.exists(os.path.join(output_folder, f"{tag}.pdb"))
                ]

            if not tags_to_extract:
                continue

            command = f"{executable} -silent {silent_file} -tags " + " ".join(tags_to_extract)
            for param_file in extra_res_files:
                command += f" -extra_res_fa {param_file}"
            command += " 2>/dev/null"

            cwd = os.getcwd()
            try:
                os.chdir(output_folder)
                exit_code = os.system(command)
            finally:
                os.chdir(cwd)

            for tag in tags_to_extract:
                pdb_path = os.path.join(output_folder, f"{tag}.pdb")
                if exit_code == 0 and os.path.exists(pdb_path):
                    models.append(pdb_path)
                elif not os.path.exists(pdb_path):
                    print(f"Failed to extract pose '{tag}' from {silent_file}.")

        self.getModelsSequences()

        if missing_models:
            print("Missing models in Rosetta folder:")
            print("\t" + ", ".join(missing_models))

        return models

    def convertLigandPDBtoMae(self, ligands_folder, change_ligand_name=True, keep_pdbs=False):
        """
        Convert ligand PDBs into MAE files.

        Parameters
        ==========
        ligands_folder : str
            Path to the folder where ligands are in PDB format
        """

        _copyScriptFile(ligands_folder, "PDBtoMAE.py")
        script_name = "._PDBtoMAE.py"

        cwd = os.getcwd()
        os.chdir(ligands_folder)
        command = "run ._PDBtoMAE.py"
        if change_ligand_name:
            command += " --change_ligand_name"
        if keep_pdbs:
            command += ' --keep_pdbs'
        os.system(command)
        os.chdir(cwd)

    def convertLigandMAEtoPDB(self, ligands_folder, change_ligand_name=True, modify_maes=False,
                              assign_pdb_names=False):
        """
        Convert ligand MAEs into PDB files.

        Parameters
        ==========
        ligands_folder : str
            Path to the folder where ligands are in MAE format
        """

        if modify_maes:
            keep_maes = True

        if isinstance(change_ligand_name, dict):
            with open(ligands_folder+'/ligand_names.json', 'w') as jf:
                json.dump(change_ligand_name, jf)
        elif isinstance(change_ligand_name, str):
            if len(change_ligand_name) != 3:
                raise ValueError('The ligand name should be three-letters long')

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(ligands_folder, "MAEtoPDB.py")
        script_name = "._MAEtoPDB.py"

        cwd = os.getcwd()
        os.chdir(ligands_folder)
        command = "run ._MAEtoPDB.py"
        if isinstance(change_ligand_name, dict):
            command += " --residue_names ligand_names.json"
        elif isinstance(change_ligand_name, str):
            command += " --residue_names "+change_ligand_name
        if change_ligand_name:
            command += " --change_ligand_name"
        if modify_maes:
            command += ' --modify_maes'
        if assign_pdb_names:
            command += ' --assign_pdb_names'
        os.system(command)
        os.chdir(cwd)

        if isinstance(change_ligand_name, dict):
            os.remove(ligands_folder+'/ligand_names.json')

    def getDockingDistances(self, model, ligand):
        """
        Get the distances related to a model and ligand docking.
        """
        distances = []

        if model not in self.docking_distances:
            return None

        if ligand not in self.docking_distances[model]:
            return None

        for d in self.docking_distances[model][ligand]:
            distances.append(d)

        if distances != []:
            return distances
        else:
            return None

    def calculateModelsDistances(self, atom_pairs, verbose=False):
        """
        Calculate models distances for a set of atom pairs.

        The atom pairs must be given in a dicionary with each key representing the name
        of a model and each value a list of the atom pairs to calculate in the format:
            {model_name: [((chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name)), ...], ...}

        Paramters
        =========
        atom_pairs : dict
            Atom pairs to calculate for each model
        """

        ### Add all label entries to dictionary
        for model in self.structures:
            self.distance_data[model] = {}
            self.distance_data[model]["model"] = []

            for d in atom_pairs[model]:
                # Generate label for distance
                label = "distance_"
                label += "_".join([str(x) for x in d[0]]) + "-"
                label += "_".join([str(x) for x in d[1]])
                self.distance_data[model].setdefault(label, [])

        for model in self.structures:

            if verbose:
                print("Calculating distances for model %s" % model)

            self.distance_data[model]["model"].append(model)

            # Get atoms in atom_pairs as dictionary
            atoms = {}
            for d in atom_pairs[model]:
                for t in d:
                    atoms.setdefault(t[0], {})
                    atoms[t[0]].setdefault(t[1], [])
                    atoms[t[0]][t[1]].append(t[2])

            # Get atom coordinates for each atom in atom_pairs
            coordinates = {}
            for chain in self.structures[model].get_chains():
                if chain.id in atoms:
                    coordinates[chain.id] = {}
                    for r in chain:
                        if r.id[1] in atoms[chain.id]:
                            coordinates[chain.id][r.id[1]] = {}
                            for atom in r:
                                if atom.name in atoms[chain.id][r.id[1]]:
                                    coordinates[chain.id][r.id[1]][
                                        atom.name
                                    ] = atom.coord

            # Calculate atom distances
            for d in atom_pairs[model]:

                # Generate label for distance
                label = "distance_"
                label += "_".join([str(x) for x in d[0]]) + "-"
                label += "_".join([str(x) for x in d[1]])

                # Calculate distance
                atom1 = d[0]
                atom2 = d[1]

                if atom1[2] not in coordinates[atom1[0]][atom1[1]]:
                    raise ValueError(
                        "Atom name %s was not found in residue %s of chain %s for model %s"
                        % (atom1[2], atom1[1], atom1[0], model)
                    )
                if atom2[2] not in coordinates[atom2[0]][atom2[1]]:
                    raise ValueError(
                        "Atom name %s was not found in residue %s of chain %s for model %s"
                        % (atom2[2], atom2[1], atom2[0], model)
                    )

                coord1 = coordinates[atom1[0]][atom1[1]][atom1[2]]
                coord2 = coordinates[atom2[0]][atom2[1]][atom2[2]]
                value = np.linalg.norm(coord1 - coord2)

                # Add data to dataframe
                self.distance_data[model][label].append(value)

            self.distance_data[model] = pd.DataFrame(
                self.distance_data[model]
            ).reset_index()
            self.distance_data[model].set_index("model", inplace=True)

        return self.distance_data

    def getModelDistances(self, model):
        """
        Get the distances associated with a specific model included in the
        self.distance_data atrribute. This attribute must be calculated in advance
        by running the calculateModelsDistances() function.

        Parameters
        ==========
        model : str
            model name
        """

        model_data = self.distance_data[model]
        distances = []
        for d in model_data:
            if "distance_" in d:
                if not model_data[d].dropna().empty:
                    distances.append(d)
        return distances

    def combineModelDistancesIntoMetric(self, metric_distances, overwrite=False):
        """
        Combine different equivalent distances contained in the self.distance_data
        attribute into specific named metrics. The function takes as input a
        dictionary (catalytic_labels) composed of inner dictionaries as follows:

            catalytic_labels = {
                metric_name = {
                    protein = distances_list}}}

        The innermost distances_list object contains all equivalent distance names for
        a specific protein to be combined under the same metric_name column.

        The combination is done by taking the minimum value of all equivalent distances.

        Parameters
        ==========
        catalytic_labels : dict
            Dictionary defining which distances will be combined under a common name.
            (for details see above).
        """

        if isinstance(self.models_data, dict) and self.models_data == {}:
            self.models_data["model"] = [m for m in self]

        for name in metric_distances:

            if "metric_" + name in self.models_data.keys() and not overwrite:
                print(
                    "Combined metric %s already added. Give overwrite=True to recombine"
                    % name
                )
            else:
                values = []
                models = []

                for model in self:
                    model_data = self.distance_data[model]
                    model_distances = metric_distances[name][model]
                    values += model_data[model_distances].min(axis=1).tolist()

                self.models_data["metric_" + name] = values

        if isinstance(self.models_data, dict):
            self.models_data = pd.DataFrame(self.models_data)
            self.models_data.set_index("model", inplace=True)

        return self.models_data

    def getModelsProtonationStates(self, residues=None):
        """
        Get the protonation state of all or specific residues in all protein models.

        For getting the protonation states of only a subset of residues a dictionary must
        be given with the 'residues' option. The keys of the dictionary are the models'
        names, and, the values, lists of tuples defining each residue to be query. The
        residue's tuples are defined as: (chain_id, residue_id).

        Parameters
        ==========
        residues : dict
            Dictionary with lists of tuples of residues (e.g., (chain_id, residue_id)) to query for each model.

        Returns
        =======
        protonation_states : pandas.DataFrame
            Data frame with protonation information.
        """

        # Set input dictionary to store protonation states
        self.protonation_states = {}
        self.protonation_states["Model"] = []
        self.protonation_states["Chain"] = []
        self.protonation_states["Residue"] = []
        self.protonation_states["Name"] = []
        self.protonation_states["State"] = []

        # Iterate all models' structures
        for model in self.models_names:
            structure = self.structures[model]
            for r in structure.get_residues():

                # Skip if a list of residues is given per model
                if residues != None:
                    if (r.get_parent().id, r.id[1]) not in residues[model]:
                        continue

                # Get Histidine protonation states
                if r.resname == "HIS":
                    atoms = [a.name for a in r]

                    # Check for hydrogens
                    hydrogens = [a for a in atoms if a.startswith('H')]

                    if not hydrogens:
                        print(f'The model {model} have not been protonated.')
                        continue

                    if "HE2" in atoms and "HD1" in atoms:
                        self.protonation_states["State"].append("HIP")
                    elif "HD1" in atoms:
                        self.protonation_states["State"].append("HID")
                    elif "HE2" in atoms:
                        self.protonation_states["State"].append("HIE")
                    else:
                        print(f'HIS {r.id[1]} could not be assigned for model {model}')
                        continue
                    self.protonation_states["Model"].append(model)
                    self.protonation_states["Chain"].append(r.get_parent().id)
                    self.protonation_states["Residue"].append(r.id[1])
                    self.protonation_states["Name"].append(r.resname)

        if self.protonation_states['Model'] == []:
            raise ValueError("No protonation states were found. Did you run prepwizard?")

        # Convert dictionary to Pandas
        self.protonation_states = pd.DataFrame(self.protonation_states)
        self.protonation_states.set_index(["Model", "Chain", "Residue"], inplace=True)

        return self.protonation_states

    def combineDockingDistancesIntoMetrics(self, catalytic_labels, overwrite=False):
        """
        Combine different equivalent distances into specific named metrics. The function
        takes as input a dictionary (catalytic_labels) composed of inner dictionaries as follows:

            catalytic_labels = {
                metric_name = {
                    protein = {
                        ligand = distances_list}}}

        The innermost distances_list object contains all equivalent distance names for
        a specific protein and ligand pair to be combined under the same metric_name column.

        The combination is done by taking the minimum value of all equivalent distances.

        Parameters
        ==========
        catalytic_labels : dict
            Dictionary defining which distances will be combined under a common name.
            (for details see above).
        """

        for name in catalytic_labels:
            if "metric_" + name in self.docking_data.keys() and not overwrite:
                print(
                    "Combined metric %s already added. Give overwrite=True to recombine"
                    % name
                )
            else:
                # Initialize a Series with NaN values, indexed the same as self.docking_data
                metric_series = pd.Series(np.nan, index=self.docking_data.index)

                for model in self.docking_data.index.get_level_values("Protein").unique():
                    # Check whether model is found in docking distances
                    if model not in self.docking_distances:
                        continue

                    model_series = self.docking_data.xs(model, level="Protein")

                    for ligand in model_series.index.get_level_values("Ligand").unique():
                        # Check whether ligand is found in model's docking distances
                        if ligand not in self.docking_distances[model]:
                            continue

                        ligand_series = model_series.xs(ligand, level="Ligand")

                        # Check input metric
                        distance_metric = False
                        angle_metric = False
                        for x in catalytic_labels[name][model][ligand]:
                            if len(x.split("-")) == 2:
                                distance_metric = True
                            elif len(x.split("-")) == 3:
                                angle_metric = True

                        if distance_metric and angle_metric:
                            raise ValueError(
                                f"Metric {name} combines distances and angles which is not supported."
                            )

                        if distance_metric:
                            distances = catalytic_labels[name][model][ligand]
                            distance_values = self.docking_distances[model][ligand][distances].min(axis=1)
                            # Align the indices
                            indices = ligand_series.index
                            metric_series.loc[(model, ligand, indices)] = distance_values.values
                            self.docking_metric_type[name] = "distance"
                        elif angle_metric:
                            angles = catalytic_labels[name][model][ligand]
                            if len(angles) > 1:
                                raise ValueError(
                                    "Combining more than one angle into a metric is not currently supported."
                                )
                            angle_values = self.docking_angles[model][ligand][angles].min(axis=1)
                            indices = ligand_series.index
                            metric_series.loc[(model, ligand, indices)] = angle_values.values
                            self.docking_metric_type[name] = "angle"

                # Assign the Series to the DataFrame
                self.docking_data["metric_" + name] = metric_series

    def combineDockingMetricsWithExclusions(self, combinations, exclusions, drop=True):
        """
        Combine mutually exclusive metrics into new metrics while handling exclusions.

        Parameters
        ----------
        combinations : dict
            Dictionary defining which metrics to combine under a new common name.
            Structure:
                combinations = {
                    new_metric_name: (metric1, metric2, ...),
                    ...
                }

        exclusions : list of tuples or dict
            List of tuples (for simple exclusions) or dictionary by metrics for by-metric exclusions.

        drop : bool, optional
            If True, drop the original metric columns after combining. Default is True.

        """

        # Determine exclusion type
        simple_exclusions = False
        by_metric_exclusions = False
        if isinstance(exclusions, list):
            simple_exclusions = True
        elif isinstance(exclusions, dict):
            by_metric_exclusions = True
        else:
            raise ValueError('exclusions should be a list of tuples or a dictionary by metrics.')

        # Collect all unique metrics from combinations
        unique_metrics = set()
        for new_metric, metrics in combinations.items():
            metric_types = [self.docking_metric_type[m] for m in metrics]
            if len(set(metric_types)) != 1:
                raise ValueError('Attempting to combine different metric types (e.g., distances and angles) is not allowed.')
            self.docking_metric_type[new_metric] = metric_types[0]
            unique_metrics.update(metrics)

        # Build a mapping from metric names to column indices
        metrics_list = list(unique_metrics)
        metrics_indexes = {m: idx for idx, m in enumerate(metrics_list)}

        # Add metric prefix if not given
        add_metric_prefix = True
        for m in metrics_list:
            if 'metric_' in m:
                raise ValueError('"metric_" prefix found in given metrics. Please, leave it out.')
        all_metrics_columns = ['metric_' + m for m in metrics_list]

        # Ensure all required metric columns exist in the data
        missing_columns = set(all_metrics_columns) - set(self.docking_data.columns)
        if missing_columns:
            raise ValueError(f"Missing metric columns in data: {missing_columns}")

        # Extract metric data
        data = self.docking_data[all_metrics_columns]

        # Positions of values to be excluded (row index, column index)
        excluded_positions = set()

        # Get labels of the shortest distance for each row
        min_metric_labels = data.idxmin(axis=1)  # Series of column names

        if simple_exclusions:
            for row_idx, metric_col_label in enumerate(min_metric_labels):
                m = metric_col_label.replace('metric_', '')

                # Exclude metrics specified in exclusions
                for exclusion_group in exclusions:
                    if m in exclusion_group:
                        others = set(exclusion_group) - {m}
                        for x in others:
                            if x in metrics_indexes:
                                col_idx = metrics_indexes[x]
                                excluded_positions.add((row_idx, col_idx))

                # Exclude other metrics in the same combination group
                for metrics_group in combinations.values():
                    if m in metrics_group:
                        others = set(metrics_group) - {m}
                        for y in others:
                            if y in metrics_indexes:
                                col_idx = metrics_indexes[y]
                                excluded_positions.add((row_idx, col_idx))

        if by_metric_exclusions:
            # Convert data to a NumPy array for efficient processing
            data_array = data.to_numpy()

            # Iterate over each row to handle exclusions iteratively
            for row_idx in range(data_array.shape[0]):

                considered_metrics = set()  # Track metrics already considered as minimums in this row

                while True:
                    # Find the minimum among metrics that haven't been excluded or considered as minimums
                    min_value = np.inf
                    min_col_idx = -1

                    # Identify the next lowest metric that hasn't been excluded or already considered
                    for col_idx, metric_value in enumerate(data_array[row_idx]):
                        if col_idx not in considered_metrics and (row_idx, col_idx) not in excluded_positions:
                            if metric_value < min_value:
                                min_value = metric_value
                                min_col_idx = col_idx
                    # if row_idx == 3:
                        # print(min_value, min_col_idx, data.columns[min_col_idx])

                    # Break the loop if no valid minimum metric is found
                    if min_col_idx == -1:
                        break

                    # Mark this metric as considered so it's not reused as minimum in future iterations
                    considered_metrics.add(min_col_idx)

                    # Get the name of the metric and retrieve exclusions based on this metric
                    min_metric_label = data.columns[min_col_idx]
                    min_metric_name = min_metric_label.replace('metric_', '')
                    excluded_metrics = exclusions.get(min_metric_name, [])

                    # Apply exclusions for this metric
                    for excluded_metric in excluded_metrics:
                        if excluded_metric in metrics_indexes:
                            excluded_col_idx = metrics_indexes[excluded_metric]
                            if (row_idx, excluded_col_idx) not in excluded_positions:
                                excluded_positions.add((row_idx, excluded_col_idx))
                                data_array[row_idx, excluded_col_idx] = np.inf  # Set excluded metric to infinity
                # if row_idx == 3:
                #     print()
                #     for x, m in zip(data_array[row_idx], metrics_indexes.items()):
                #         print(x, m)

        # Combine metrics and add new columns to the DataFrame
        for new_metric_name, metrics_to_combine in combinations.items():
            c_indexes = [metrics_indexes[m] for m in metrics_to_combine if m in metrics_indexes]

            if c_indexes:
                # Calculate the minimum value among the combined metrics, excluding inf-only combinations
                combined_min = np.min(data_array[:, c_indexes], axis=1)

                # Check if combined_min is all inf and handle accordingly
                if np.all(np.isinf(combined_min)):
                    print(f"Skipping combination for '{new_metric_name}' due to incompatible exclusions.")
                    continue
                self.docking_data['metric_' + new_metric_name] = combined_min
            else:
                raise ValueError(f"No valid metrics to combine for '{new_metric_name}'.")

        # Drop original metric columns if specified
        if drop:
            self.docking_data.drop(columns=all_metrics_columns, inplace=True)

        # Ensure compatibility of combinations with exclusions
        for new_metric_name, metrics_to_combine in combinations.items():
            non_excluded_found = False

            for metric in metrics_to_combine:
                # Use standardized names for consistent indexing
                metric_column_name = 'metric_' + metric if 'metric_' not in metric else metric
                col_idx = metrics_indexes.get(metric_column_name)

                if col_idx is not None:
                    # Check directly in data_array for non-excluded values
                    column_values = data_array[:, col_idx]
                    if not np.all(np.isinf(column_values)):
                        non_excluded_found = True
                        break

            # Print warning if all values for a combination are excluded
            if not non_excluded_found:
                print(f"Warning: No non-excluded metrics available to combine for '{new_metric_name}'.")

    def plotDockingData(self):
        """
        Generates an interactive scatter plot for docking data, allowing users to select
        the protein, ligand, and columns for the X and Y axes.

        The method assumes the docking data is a Pandas DataFrame stored in `self.docking_data`
        with a MultiIndex (Protein, Ligand, Pose) and numeric columns (Score, RMSD, Closest distance).

        The function creates interactive widgets to select a specific protein, ligand, and which
        numeric columns to plot on the X and Y axes.

        Returns:
            An interactive scatter plot that updates based on widget selections.
        """

        # Subfunction to handle filtering and plotting
        def scatter_plot(protein, ligand, x_axis, y_axis):
            """
            Subfunction to plot the scatter plot for the selected protein and ligand.

            Args:
                protein (str): Selected protein sequence.
                ligand (str): Selected ligand.
                x_axis (str): The column name for the X-axis.
                y_axis (str): The column name for the Y-axis.
            """
            # Filter the data based on selected Protein and Ligand
            filtered_df = df.loc[(protein, ligand)]

            # Plotting the scatter plot for the selected X and Y axes
            plt.figure(figsize=(8, 6))
            plt.scatter(filtered_df[x_axis], filtered_df[y_axis], color='blue')
            plt.title(f'Scatter Plot for {protein} - {ligand}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.grid(True)
            plt.show()

        # Get the docking data from the object's attribute
        df = self.docking_data

        # Create dropdown widgets for selecting the Protein, Ligand, X-axis, and Y-axis columns
        protein_dropdown = widgets.Dropdown(
            options=df.index.levels[0],  # Options for Protein selection
            description='Protein'        # Label for the dropdown
        )

        ligand_dropdown = widgets.Dropdown(
            options=df.index.levels[1],  # Options for Ligand selection
            description='Ligand'         # Label for the dropdown
        )

        x_axis_dropdown = widgets.Dropdown(
            options=df.columns,          # Options for selecting the X-axis (numeric columns)
            description='X-axis'         # Label for the dropdown
        )

        y_axis_dropdown = widgets.Dropdown(
            options=df.columns,          # Options for selecting the Y-axis (numeric columns)
            description='Y-axis'         # Label for the dropdown
        )

        # Create an interactive widget that dynamically updates the plot based on selections
        interact(
            scatter_plot,
            protein=protein_dropdown,
            ligand=ligand_dropdown,
            x_axis=x_axis_dropdown,
            y_axis=y_axis_dropdown
        )

    def getBestDockingPoses(
        self,
        filter_values,
        n_models=1,
        return_failed=False,
        exclude_models=None,
        exclude_ligands=None,
        exclude_pairs=None,
    ):
        if exclude_models is None:
            exclude_models = []
        if exclude_ligands is None:
            exclude_ligands = []
        if exclude_pairs is None:
            exclude_pairs = []

        if not isinstance(n_models, int):
            n_models = int(n_models)

        # Create exclusion masks
        docking_data = self.docking_data
        index = docking_data.index

        exclude_models_mask = ~index.get_level_values('Protein').isin(exclude_models)
        exclude_ligands_mask = ~index.get_level_values('Ligand').isin(exclude_ligands)

        pairs_to_exclude = set(exclude_pairs)
        if pairs_to_exclude:
            exclude_pairs_mask = ~index.map(lambda idx: (idx[0], idx[1]) in pairs_to_exclude)
        else:
            exclude_pairs_mask = np.ones(len(index), dtype=bool)  # Include all

        mask = exclude_models_mask & exclude_ligands_mask & exclude_pairs_mask

        filtered_data = docking_data[mask]

        # Apply filters
        for metric in filter_values:
            filter_value = filter_values[metric]
            if metric not in ["Score", "RMSD"]:
                if not metric.startswith("metric_") and metric != 'Closest distance':
                    metric_label = "metric_" + metric
                else:
                    metric_label = metric
            else:
                metric_label = metric

            if isinstance(filter_value, (float, int)):
                filtered_data = filtered_data[filtered_data[metric_label] <= filter_value]
            elif isinstance(filter_value, (tuple, list)):
                filtered_data = filtered_data[
                    (filtered_data[metric_label] >= filter_value[0]) &
                    (filtered_data[metric_label] <= filter_value[1])
                ]
            else:
                filtered_data = filtered_data[filtered_data[metric_label] < filter_value]

        # Ensure index levels are named
        if filtered_data.index.nlevels == 3:
            filtered_data.index.set_names(['Protein', 'Ligand', 'Pose'], inplace=True)
        else:
            # If index levels are not named, we can set default names
            filtered_data.index.set_names(['Protein', 'Ligand'], inplace=True)

        # Get all available pairs after exclusions
        available_pairs = docking_data[mask].index.to_frame(index=False)[['Protein', 'Ligand']].drop_duplicates()

        # Get pairs present in filtered_data
        filtered_pairs = filtered_data.index.to_frame(index=False)[['Protein', 'Ligand']].drop_duplicates()

        # Find failed pairs
        failed_pairs = pd.merge(
            available_pairs,
            filtered_pairs,
            on=['Protein', 'Ligand'],
            how='left',
            indicator=True
        )
        failed_pairs = failed_pairs[failed_pairs['_merge'] == 'left_only'][['Protein', 'Ligand']]
        failed = list(failed_pairs.itertuples(index=False, name=None))

        # Sort and group
        filtered_data = filtered_data.sort_values(by=['Protein', 'Ligand', 'Score'])

        # Use level indices if names are not consistent
        if filtered_data.index.nlevels >= 2:
            grouped = filtered_data.groupby(level=[0, 1], sort=False)
        else:
            grouped = filtered_data.groupby(['Protein', 'Ligand'], sort=False)

        # Select top n_models per group
        top_n = grouped.head(n_models)

        # Warning for groups with less than n_models
        group_sizes = grouped.size()
        # print("Group Sizes:")
        # print(group_sizes)
        # print("Data Types of Group Sizes:")
        # print(group_sizes.dtypes)
        if not group_sizes.empty:
            insufficient_groups = group_sizes[group_sizes < n_models]
            if not insufficient_groups.empty:
                for (protein, ligand), size in insufficient_groups.iteritems():
                    print(
                        "WARNING: less than %s models available for docking %s + %s"
                        % (n_models, protein, ligand)
                    )
        else:
            insufficient_groups = pd.Series(dtype=int)

        if return_failed:
            return failed, top_n
        else:
            return top_n

    def getBestDockingPosesIteratively(
        self,
        metrics,
        ligands=None,
        distance_step=0.1,
        angle_step=1.0,
        fixed=None,
        max_distance=None,
        max_distance_step_shift=None,
        verbose=False,
    ):
        """
        Iteratively select the best docking poses for protein-ligand pairs based on given metric thresholds.
        If not all protein-ligand pairs have acceptable models under the initial thresholds, the function
        iteratively relaxes the thresholds of the non-fixed metrics, starting with the ones that accept the
        fewest models, until at least one model is selected for each protein-ligand pair or until
        max_distance is reached.

        Parameters:
        - metrics (dict): Dictionary of metric thresholds. Keys are metric names, values are thresholds.
                          Thresholds can be a scalar (upper limit) or a tuple/list (lower and upper limits).
        - ligands (list, optional): List of ligands to consider. If None, all ligands are considered.
        - distance_step (float, optional): Step size to adjust distance metrics.
        - angle_step (float, optional): Step size to adjust angle metrics.
        - fixed (list, optional): List of metric names that should not be adjusted.
        - max_distance (float, optional): Maximum allowed value for distance metrics.
        - max_distance_step_shift (float, optional): New step size for distance metrics after reaching max_distance.

        Returns:
        - pandas.DataFrame: DataFrame containing the selected docking poses.
        """

        # Ensure fixed is a list
        if fixed is None:
            fixed = []
        elif isinstance(fixed, str):
            fixed = [fixed]

        # Ensure there is at least one non-fixed metric
        non_fixed_metrics = set(metrics.keys()) - set(fixed)
        if not non_fixed_metrics:
            raise ValueError("You must leave at least one metric not fixed")

        metrics = metrics.copy()

        # Filter data by ligands if provided
        if ligands is not None:
            # Assuming that the ligand identifier is at index level 1
            data = self.docking_data[self.docking_data.index.get_level_values(1).isin(ligands)]
        else:
            data = self.docking_data

        # Get all unique protein-ligand pairs
        protein_ligand_pairs = set(zip(
            data.index.get_level_values(0),  # Assuming protein identifier is at index level 0
            data.index.get_level_values(1)   # Ligand identifier at index level 1
        ))

        extracted_pairs = set()
        selected_indexes = []
        current_distance_step = distance_step
        step_shift_applied = False  # Flag to indicate if step shift has been applied

        while len(extracted_pairs) < len(protein_ligand_pairs):
            if verbose:
                ti = time.time()

            # Get best poses with current thresholds
            best_poses = self.getBestDockingPoses(metrics, n_models=1)  # Assuming self has this method

            # Select new models
            new_selected_pairs = set()
            for idx in best_poses.index:
                pair = (idx[0], idx[1])  # Adjust index levels if needed
                if pair not in extracted_pairs:
                    selected_indexes.append(idx)
                    new_selected_pairs.add(pair)

            extracted_pairs.update(new_selected_pairs)

            # If we've selected models for all pairs, break the loop
            if len(extracted_pairs) >= len(protein_ligand_pairs):
                break

            # Prepare remaining data
            remaining_pairs = protein_ligand_pairs - extracted_pairs
            mask = [((idx[0], idx[1]) in remaining_pairs) for idx in data.index]
            remaining_data = data[mask]

            if remaining_data.empty:
                break  # No more data to process

            # Compute acceptance counts for each metric
            metric_acceptance = {}
            for metric in metrics:
                if metric in fixed:
                    continue
                metric_label = metric if metric.startswith('metric_') else 'metric_' + metric
                metric_type = self.docking_metric_type.get(metric_label.replace('metric_', ''), None)
                if metric_type is None:
                    raise ValueError(f"Metric type for {metric_label} not defined.")

                metric_values = remaining_data[metric_label]

                if isinstance(metrics[metric], (int, float)):
                    if metric_type in ['distance', 'angle']:
                        acceptance = metric_values <= metrics[metric]
                    else:
                        acceptance = metric_values >= metrics[metric]
                elif isinstance(metrics[metric], (tuple, list)):
                    lower, upper = metrics[metric]
                    acceptance = (metric_values >= lower) & (metric_values <= upper)
                else:
                    raise ValueError(f"Invalid threshold type for metric {metric}")

                metric_acceptance[metric] = acceptance.sum()

            # Order metrics by acceptance count (ascending)
            ordered_metrics = sorted(
                [(m, a) for m, a in metric_acceptance.items() if m not in fixed],
                key=lambda x: x[1]
            )

            # Adjust thresholds for the metric with lowest acceptance
            updated = False
            for metric, _ in ordered_metrics:
                metric_label = metric if metric.startswith('metric_') else 'metric_' + metric
                metric_type = self.docking_metric_type.get(metric_label.replace('metric_', ''), None)
                if metric_type == 'distance':
                    step = current_distance_step
                elif metric_type == 'angle':
                    step = angle_step
                else:
                    raise ValueError(f"Unknown metric type for {metric_label}")

                if isinstance(metrics[metric], (int, float)):
                    # For upper limit thresholds (assuming distance and angle are upper limits)
                    new_value = metrics[metric] + step

                    if metric_type == 'distance' and max_distance is not None:
                        if not step_shift_applied and new_value >= max_distance:
                            if max_distance_step_shift is not None:
                                # Apply step shift
                                current_distance_step = max_distance_step_shift
                                step_shift_applied = True
                                print(f"Max distance {max_distance} reached for metric {metric}. Applying step shift to {current_distance_step}.")
                                # Do not cap the value; allow it to exceed max_distance
                            else:
                                # If no step shift, cap at max_distance and terminate
                                new_value = max_distance
                                metrics[metric] = new_value
                                print(f"Max distance {max_distance} reached for metric {metric}. Terminating iteration.")
                                updated = True
                                break  # Exit the for-loop to terminate the while-loop
                    # Update the metric
                    metrics[metric] = new_value
                    updated = True
                    break  # Adjusted one metric, exit the loop

                elif isinstance(metrics[metric], (tuple, list)):
                    # For range thresholds
                    lower, upper = metrics[metric]
                    new_lower = lower - step
                    new_upper = upper + step
                    metrics[metric] = (new_lower, new_upper)
                    updated = True
                    break  # Adjusted one metric, exit the loop
                else:
                    raise ValueError(f"Invalid threshold type for metric {metric}")

            # Check if step shift was applied and allow further adjustments
            if not updated:
                # Could not adjust any metric, exit the loop
                print("No metrics were updated. Terminating iteration.")
                break

            # If step shift was applied and already applied before, continue adjusting with new step size
            # No additional action needed as current_distance_step has been updated

            # Optional: Print progress for debugging
            if verbose:
                elapsed_time = time.time() - ti
                print(f"Max distance reached: {step_shift_applied}, Current step: {current_distance_step}, Metrics: {metrics}, Time elapsed: {elapsed_time:.2f}s", end='\r')

        # Collect selected models
        if selected_indexes:
            best_poses = data.loc[selected_indexes]
        else:
            best_poses = pd.DataFrame()  # Return empty DataFrame if no poses selected

        return best_poses

    def extractDockingPoses(
        self,
        docking_data,
        docking_folder,
        output_folder,
        separator="-",
        only_extract_new=True,
        covalent_check=True,
        remove_previous=False,
    ):
        """
        Extract docking poses present in a docking_data dataframe. The docking DataFrame
        contains the same structure as the self.docking_data dataframe, parameter of
        this class. This dataframe makes reference to the docking_folder where the
        docking results are contained.

        Parameters
        ==========
        dockign_data : pandas.DataFrame
            Datframe containing the poses to be extracted
        docking_folder : str
            Path the folder containing the docking results
        output_folder : str
            Path to the folder where the docking structures will be saved.
        separator : str
            Symbol used to separate protein, ligand, and docking pose index.
        only_extract_new : bool
            Only extract models not present in the output_folder
        remove_previous : bool
            Remove all content in the output folder
        """

        # Check the separator is not in model or ligand names
        for model in self.docking_ligands:
            if separator in str(model):
                raise ValueError(
                    "The separator %s was found in model name %s. Please use a different separator symbol."
                    % (separator, model)
                )
            for ligand in self.docking_ligands[model]:
                if separator in ligand:
                    raise ValueError(
                        "The separator %s was found in ligand name %s. Please use a different separator symbol."
                        % (separator, ligand)
                    )

        # Remove output_folder
        if os.path.exists(output_folder):
            if remove_previous:
                shutil.rmtree(output_folder)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        else:
            # Gather already extracted models
            if only_extract_new:
                extracted_models = set()
                for model in os.listdir(output_folder):
                    if not os.path.isdir(output_folder + "/" + model):
                        continue
                    for f in os.listdir(output_folder + "/" + model):
                        if f.endswith(".pdb"):
                            m, l = f.split(separator)[:2]
                            extracted_models.add((m, l))

                # Filter docking data to not include the already extracted models
                extracted_indexes = []
                for i in docking_data.index:
                    if i[:2] in extracted_models:
                        extracted_indexes.append(i)
                docking_data = docking_data[~docking_data.index.isin(extracted_indexes)]
                if docking_data.empty:
                    print("All models were already extracted!")
                    print("Set only_extract_new=False to extract them again!")
                    return
                else:
                    if len(extracted_models):
                        print(f"{len(extracted_models)} models were already extracted!")
                    print(f"Extracting {docking_data.shape[0]} new models")

        # Copy analyse docking script (it depends on schrodinger so we leave it out.)
        _copyScriptFile(output_folder, "extract_docking.py")
        script_path = output_folder + "/._extract_docking.py"

        # Move to output folder
        os.chdir(output_folder)

        # Save given docking data to csv
        dd = docking_data.reset_index()
        dd.to_csv("._docking_data.csv", index=False)

        # Execute docking analysis
        command = (
            "run ._extract_docking.py ._docking_data.csv ../"
            + docking_folder
            + " --separator "
            + separator
        )
        os.system(command)

        # Remove docking data
        os.remove("._docking_data.csv")

        # move back to folder
        os.chdir("..")

        # Check models for covalent residues
        for protein in os.listdir(output_folder):
            if not os.path.isdir(output_folder + "/" + protein):
                continue
            for f in os.listdir(output_folder + "/" + protein):
                if covalent_check:
                    self._checkCovalentLigands(
                        protein,
                        output_folder + "/" + protein + "/" + f,
                        check_file=True,
                    )

    def getSingleDockingData(self, protein, ligand, data_frame=None):
        """
        Get the docking data for a particular combination of protein and ligand

        Parameters
        ==========
        protein : str
            Protein model name
        ligad : str
            Ligand name
        data_frame : pandas.DataFrame
            Optional dataframe to get docking data from.
        """

        if ligand not in self.docking_ligands[protein]:
            raise ValueError("has no docking data")

        if isinstance(data_frame, type(None)):
            data_frame = self.docking_data

        protein_series = data_frame[
            data_frame.index.get_level_values("Protein") == protein
        ]
        ligand_series = protein_series[
            protein_series.index.get_level_values("Ligand") == ligand
        ]

        return ligand_series

    def plotDocking(
        self,
        protein,
        ligand,
        x="RMSD",
        y="Score",
        z=None,
        colormap="Blues_r",
        output_folder=None,
        extension=".png",
        dpi=200,
    ):

        if output_folder != None:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

        protein_series = self.docking_data[
            self.docking_data.index.get_level_values("Protein") == protein
        ]
        if protein_series.empty:
            print("Model %s not found in Docking data" % protein)
            return None
        ligand_series = protein_series[
            protein_series.index.get_level_values("Ligand") == ligand
        ]
        if ligand_series.empty:
            print(
                "Ligand %s not found in Docking data for protein %s" % (ligand, protein)
            )
            return None

        fig, ax = plt.subplots()
        if z != None:
            ligand_series.plot(kind="scatter", x=x, y=y, c=z, colormap=colormap, ax=ax)
        else:
            ligand_series.plot(kind="scatter", x=x, y=y, ax=ax)

        plt.title(protein + " + " + ligand)
        if output_folder != None:
            plt.savefig(
                output_folder + "/" + protein + "_" + ligand + extension, dpi=dpi
            )
            plt.close()

    def loadModelsFromPrepwizardFolder(
        self,
        prepwizard_folder,
        return_missing=False,
        return_failed=False,
        covalent_check=True,
        models=None,
        atom_mapping=None,
        conect_update=False,
        replace_symbol=None,
        collect_memory_every=None,
        only_hetatoms_conect=False,
    ):
        """
        Read structures from a Schrodinger calculation.

        Parameters
        ==========
        prepwizard_folder : str
            Path to the output folder from a prepwizard calculation
        """

        if (
            replace_symbol
            and not isinstance(replace_symbol, tuple)
            and len(replace_symbol) != 2
        ):
            raise ValueError("replace_symbol must be a tuple: (old_symbol, new_symbol)")

        all_models = []
        failed_models = []
        load_count = 0  # For collect memory
        collect_memory = False
        for d in os.listdir(prepwizard_folder + "/output_models"):
            if os.path.isdir(prepwizard_folder + "/output_models/" + d):
                for f in os.listdir(prepwizard_folder + "/output_models/" + d):
                    if f.endswith(".log"):
                        with open(
                            prepwizard_folder + "/output_models/" + d + "/" + f
                        ) as lf:
                            for l in lf:
                                if "error" in l.lower():
                                    print(
                                        "Error was found in log file: %s. Please check the calculation!"
                                        % f
                                    )
                                    model = f.replace(".log", "")

                                    if replace_symbol:
                                        model = model.replace(
                                            replace_symbol[1], replace_symbol[0]
                                        )

                                    if models and model not in models:
                                        continue

                                    failed_models.append(model)

                                    break

                    if f.endswith(".pdb"):
                        model = f.replace(".pdb", "")

                        if replace_symbol:
                            model = model.replace(replace_symbol[1], replace_symbol[0])

                        # skip models not loaded into the library
                        if model not in self.models_names:
                            continue

                        # Skip models not in the given models list
                        if models != None and model not in models:
                            continue

                        if (
                            collect_memory_every
                            and load_count % collect_memory_every == 0
                        ):
                            collect_memory = True
                        else:
                            collect_memory = False

                        all_models.append(model)
                        self.readModelFromPDB(
                            model,
                            prepwizard_folder + "/output_models/" + d + "/" + f,
                            covalent_check=covalent_check,
                            atom_mapping=atom_mapping,
                            conect_update=conect_update,
                            collect_memory=collect_memory,
                            only_hetatoms=only_hetatoms_conect
                        )
                        load_count += 1

        self.getModelsSequences()

        # Gather missing models
        # Remove
        if models:
            missing_models = set(models) - set(all_models)
        else:
            missing_models = set(self.models_names) - set(all_models)

        if missing_models != set():
            print("Missing models in prepwizard folder:")
            print("\t" + ", ".join(missing_models))

        if return_missing:
            return missing_models
        if return_failed:
            return failed_models

    def analyseRosettaCalculation(
        self,
        rosetta_folder,
        atom_pairs=None,
        energy_by_residue=False,
        interacting_residues=False,
        query_residues=None,
        overwrite=False,
        protonation_states=False,
        decompose_bb_hb_into_pair_energies=False,
        binding_energy=False,
        cpus=None,
        return_jobs=False,
        verbose=False,
        skip_finished=False,
        pyrosetta_env=None,
        param_files=None,
    ):
        """
        Analyse Rosetta calculation folder. The analysis reads the energies and calculate distances
        between atom pairs given. Optionally the analysis get the energy of each residue in each pose.
        Additionally, it can analyse the interaction between specific residues (query_residues option)and
        their neighbouring sidechains by mutating the neighbour residues to glycines.

        The atom pairs must be given in a dicionary with each key representing the name
        of a model and each value a list of the atom pairs to calculate in the format:
            {model_name: [((chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name)), ...], ...}

        The main analysis is stored at self.rosetta_data
        The energy by residue analysis is soterd at self.rosetta_ebr_data
        Sidechain interaction analysis is stored at self.rosetta_interacting_residues

        Data is also stored in csv files inside the Rosetta folder for easy retrieving the data if found:

        The main analysis is stored at ._rosetta_data.csv
        The energy by residue analysis is soterd at ._rosetta_energy_residue_data.csv
        Sidechain interaction analysis is stored at ._rosetta_interacting_residues_data.csv


        The overwrite option forces recalcualtion of the data.

        Parameters
        ==========
        rosetta_folder : str
            Path to the Rosetta Calculation Folder.
        atom_pairs : dict
            Pairs of atom to calculate for each model.
        energy_by_residue : bool
            Calculate energy by residue data?
        overwrite : bool
            Force the data calculation from the files.
        interacting_residues : str
            Calculate interacting energies between residues
        query_residues : list
            Residues to query neoghbour atoms. Leave None for all residues (not recommended, too slow!)
        decompose_bb_hb_into_pair_energies : bool
            Store backbone hydrogen bonds in the energy graph on a per-residue basis (this doubles the
            number of calculations, so is off by default).
        binding_energy : str
            Comma-separated list of chains for which calculate the binding energy.
        pyrosetta_env : str, optional
            Name of the conda environment to activate before invoking the PyRosetta script.
        param_files : Union[str, Sequence[str]], optional
            Additional Rosetta params files or directories containing params files to
            mirror into `docking_folder/params` before running the PyRosetta analysis.
        """

        if not os.path.exists(rosetta_folder):
            raise ValueError(
                'The Rosetta calculation folder: "%s" does not exists!' % rosetta_folder
            )

        if param_files is not None:
            if isinstance(param_files, (str, os.PathLike)):
                param_sources = [param_files]
            else:
                param_sources = list(param_files)

            resolved_params: List[str] = []
            for source in param_sources:
                if os.path.isdir(source):
                    for entry in os.listdir(source):
                        if entry.endswith(".params"):
                            resolved_params.append(os.path.join(source, entry))
                else:
                    resolved_params.append(source)

            params_folder = os.path.join(rosetta_folder, "params")
            os.makedirs(params_folder, exist_ok=True)
            for param_path in resolved_params:
                abs_param = os.path.abspath(param_path)
                if not os.path.exists(abs_param):
                    raise FileNotFoundError(f"Parameter file not found: {abs_param}")
                destination = os.path.join(params_folder, os.path.basename(abs_param))
                if os.path.abspath(destination) != abs_param:
                    shutil.copyfile(abs_param, destination)

        atom_pairs_payload = None
        # Write atom_pairs dictionary to json file
        if atom_pairs is not None:
            atom_pairs_payload = _normalize_relax_atom_pairs(atom_pairs, rosetta_folder)
            with open(rosetta_folder + "/._atom_pairs.json", "w") as jf:
                json.dump(atom_pairs_payload, jf)

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(rosetta_folder, "analyse_calculation.py", subfolder="pyrosetta")

        # Execute docking analysis
        command = (
            "python "
            + rosetta_folder
            + "/._analyse_calculation.py "
            + rosetta_folder
            + " "
        )

        if binding_energy:
            command += "--binding_energy " + binding_energy + " "
        if atom_pairs_payload is not None:
            command += "--atom_pairs " + rosetta_folder + "/._atom_pairs.json "
        if return_jobs:
            command += "--models MODEL "
        if energy_by_residue:
            command += "--energy_by_residue "
        if interacting_residues:
            command += "--interacting_residues "
            if query_residues != None:
                command += "--query_residues "
                command += ",".join([str(r) for r in query_residues]) + " "
        if protonation_states:
            command += "--protonation_states "
        if decompose_bb_hb_into_pair_energies:
            command += "--decompose_bb_hb_into_pair_energies "
        if cpus != None:
            command += "--cpus " + str(cpus) + " "
        if verbose:
            command += "--verbose "
        if overwrite:
            command += "--overwrite "

        # Compile individual models for each job
        if return_jobs:
            commands = []
            for m in self:

                if not os.path.exists(f'{rosetta_folder}/output_models/{m}/{m}_relax.out'):
                    print(f'Silent file for model {m} was not found!')
                    continue

                if skip_finished and not overwrite and os.path.exists(f'{rosetta_folder}/.analysis/scores/{m}.csv'):
                    continue

                job_command = command.replace("MODEL", m)
                if pyrosetta_env:
                    job_command = _wrap_pyrosetta_command(job_command, pyrosetta_env)
                commands.append(job_command)

            print("Returning jobs for running the analysis in parallel.")
            print(
                "After jobs have finished, rerun this function removing return_jobs=True!"
            )
            return commands

        else:
            count = 0
            for m in self:
                if not os.path.exists(f'{rosetta_folder}/output_models/{m}/{m}_relax.out'):
                    print(f'Silent file for model {m} was not found!')
                    continue
                if not os.path.exists(f'{rosetta_folder}/.analysis/scores/{m}.csv'):
                    count += 1

            if count or overwrite:
                if not pyrosetta_env:
                    installed = {pkg.key for pkg in pkg_resources.working_set}
                    if 'pyrosetta' not in installed:
                        raise ValueError('PyRosetta was not found in your Python environment.\
                        Consider using return_jobs=True or activating an environment the does have it.')
                exec_command = command
                if pyrosetta_env:
                    exec_command = _wrap_pyrosetta_command(exec_command, pyrosetta_env)
                os.system(exec_command)

        # Compile dataframes into rosetta_data attributes
        self.rosetta_data = []
        self.rosetta_distances = {}
        self.rosetta_ebr = []
        self.rosetta_neighbours = []
        self.rosetta_protonation = []
        binding_energy_df = []

        output_folder = '.analysis'
        analysis_folder = rosetta_folder + '/'+output_folder
        for model in self:

            # Read scores
            scores_folder = analysis_folder + "/scores"
            scores_csv = scores_folder + "/" + model + ".csv"
            if os.path.exists(scores_csv):
                self.rosetta_data.append(pd.read_csv(scores_csv))

            # Read binding energies
            be_folder = analysis_folder + "/binding_energy"
            be_csv = be_folder + "/" + model + ".csv"
            if os.path.exists(be_csv):
                binding_energy_df.append(pd.read_csv(be_csv))

            # Read distances
            distances_folder = analysis_folder + "/distances"
            distances_csv = distances_folder + "/" + model + ".csv"
            if os.path.exists(distances_csv):
                self.rosetta_distances[model] = pd.read_csv(distances_csv)
                self.rosetta_distances[model].set_index(["Model", "Pose"], inplace=True)

            # Read energy-by-residue data
            ebr_folder = analysis_folder + "/ebr"
            erb_csv = ebr_folder + "/" + model + ".csv"
            if os.path.exists(erb_csv):
                self.rosetta_ebr.append(pd.read_csv(erb_csv))

            # Read interacting neighbours data
            neighbours_folder = analysis_folder + "/neighbours"
            neighbours_csv = neighbours_folder + "/" + model + ".csv"
            if os.path.exists(neighbours_csv):
                self.rosetta_neighbours.append(pd.read_csv(neighbours_csv))

            # Read protonation data
            protonation_folder = analysis_folder + "/protonation"
            protonation_csv = protonation_folder + "/" + model + ".csv"
            if os.path.exists(protonation_csv):
                self.rosetta_protonation.append(pd.read_csv(protonation_csv))

        if self.rosetta_data == []:
            raise ValueError("No rosetta output was found in %s" % rosetta_folder)

        self.rosetta_data = pd.concat(self.rosetta_data)
        self.rosetta_data.set_index(["Model", "Pose"], inplace=True)

        if binding_energy:

            binding_energy_df = pd.concat(binding_energy_df)
            binding_energy_df.set_index(["Model", "Pose"], inplace=True)

            # Add interface scores to rosetta_data
            for score in binding_energy_df:
                index_value_map = {}
                for i, v in binding_energy_df.iterrows():
                    index_value_map[i] = v[score]

                values = []
                for i in self.rosetta_data.index:
                    values.append(index_value_map[i])

                self.rosetta_data[score] = values

        if energy_by_residue and self.rosetta_ebr != []:
            self.rosetta_ebr = pd.concat(self.rosetta_ebr)
            self.rosetta_ebr.set_index(
                ["Model", "Pose", "Chain", "Residue"], inplace=True
            )
        else:
            self.rosetta_ebr = None

        if interacting_residues and self.rosetta_neighbours != []:
            self.rosetta_neighbours = pd.concat(self.rosetta_neighbours)
            self.rosetta_neighbours.set_index(
                ["Model", "Pose", "Chain", "Residue"], inplace=True
            )
        else:
            self.rosetta_neighbours = None

        if protonation_states and self.rosetta_protonation != []:
            self.rosetta_protonation = pd.concat(self.rosetta_protonation)
            self.rosetta_protonation.set_index(
                ["Model", "Pose", "Chain", "Residue"], inplace=True
            )
        else:
            self.rosetta_protonation = None

        return self.rosetta_data

    def getRosettaModelDistances(self, model):
        """
        Get all distances related to a model from the self.rosetta_data DataFrame.

        Parameters
        ==========
        model : str
            Model name

        Return
        ======
        distances : list
            Distances containing non-nan values for the model.

        """

        distances = []
        for d in self.rosetta_distances[model]:
            if d.startswith("distance_"):
                distances.append(d)

        return distances

    def combineRosettaDistancesIntoMetric(
        self, metric_labels, overwrite=False, rosetta_data=None, rosetta_distances=None
    ):
        """
        Combine different equivalent distances contained in the self.distance_data
        attribute into specific named metrics. The function takes as input a
        dictionary (metric_distances) composed of inner dictionaries as follows:

            metric_labels = {
                metric_name = {
                    model = distances_list}}}

        The innermost distances_list object contains all equivalent distance names for
        a specific protein to be combined under the same metric_name column.

        The combination is done by taking the minimum value of all equivalent distances.

        Parmeters
        =========
        metric_labels : dict
            Dictionary defining which distances will be combined under a common name.
            (for details see above).
        """

        if isinstance(rosetta_data, type(None)):
            rosetta_data = self.rosetta_data

        if isinstance(rosetta_distances, type(None)):
            rosetta_distances = self.rosetta_distances

        for name in metric_labels:
            if "metric_" + name in rosetta_data.keys() and not overwrite:
                print(
                    "Combined metric %s already added. Give overwrite=True to recombine"
                    % name
                )

            else:
                values = []
                for model in rosetta_data.index.levels[0]:
                    if isinstance(model, int):
                        model = str(model)
                    model_distances = rosetta_distances[model]
                    md = model_distances[metric_labels[name][model]]
                    values += md.min(axis=1).tolist()

                rosetta_data["metric_" + name] = values
                self._set_rosetta_metric_type(name, "distance")

    def combineRosettaMetricsWithExclusions(self, combinations, exclusions, drop=True):
        """
        Combine mutually exclusive Rosetta metrics (from :meth:`analyseRosettaCalculation`)
        into aggregate metrics while respecting exclusion rules.

        Parameters
        ----------
        combinations : dict
            Mapping of ``new_metric_name -> (metricA, metricB, ...)`` without the ``metric_`` prefix.
        exclusions : list or dict
            Either a list of tuples describing mutually exclusive metric names, or a dictionary
            mapping a metric to the metrics that should be excluded whenever it is selected
            as the minimum value.
        drop : bool, optional
            Drop the original metric columns after combining. Default True.
        """

        if self.rosetta_data is None or self.rosetta_data.empty:
            raise ValueError(
                "Rosetta data is empty. Run analyseRosettaCalculation before combining metrics."
            )

        if not hasattr(self, "rosetta_docking_metric_type") or self.rosetta_docking_metric_type is None:
            self.rosetta_docking_metric_type = {}

        def _with_prefix(name: str) -> str:
            return name if name.startswith("metric_") else f"metric_{name}"

        def _strip_prefix(name: str) -> str:
            return name[7:] if name.startswith("metric_") else name

        if isinstance(exclusions, list):
            simple_exclusions = True
            by_metric_exclusions = False
        elif isinstance(exclusions, dict):
            simple_exclusions = False
            by_metric_exclusions = True
        else:
            raise ValueError("exclusions should be a list of tuples or a dictionary by metrics.")

        unique_metrics = set()
        for new_metric, metrics in combinations.items():
            metric_types = []
            for metric in metrics:
                metric_label = _with_prefix(metric)
                metric_type = self._get_rosetta_metric_type(metric_label)
                if metric_type is None:
                    inferred_type = None
                    if metric_label in self.rosetta_data.columns:
                        if "angle" in metric_label.lower() or "torsion" in metric_label.lower():
                            inferred_type = "angle"
                        else:
                            inferred_type = "distance"
                    if inferred_type is None:
                        raise ValueError(
                            f"Metric type for '{metric_label}' is not defined. "
                            "Populate it via combineRosettaDistancesIntoMetric or _set_rosetta_metric_type."
                        )
                    self._set_rosetta_metric_type(metric_label, inferred_type)
                    metric_type = inferred_type
                metric_types.append(metric_type)
            if len(set(metric_types)) != 1:
                raise ValueError(
                    "Attempting to combine different metric types (e.g., distances and angles) is not allowed."
                )
            self._set_rosetta_metric_type(new_metric, metric_types[0])
            unique_metrics.update(_strip_prefix(metric) for metric in metrics)

        metrics_list = list(unique_metrics)
        metrics_indexes = {m: idx for idx, m in enumerate(metrics_list)}

        for m in metrics_list:
            if "metric_" in m:
                raise ValueError('Provide metric names without the "metric_" prefix.')

        all_metrics_columns = ["metric_" + m for m in metrics_list]
        missing_columns = set(all_metrics_columns) - set(self.rosetta_data.columns)
        if missing_columns:
            raise ValueError(f"Missing metric columns in data: {missing_columns}")

        data = self.rosetta_data[all_metrics_columns]
        excluded_positions = set()
        min_metric_labels = data.idxmin(axis=1)

        if simple_exclusions:
            for row_idx, metric_col_label in enumerate(min_metric_labels):
                m = _strip_prefix(metric_col_label)

                for exclusion_group in exclusions:
                    canonical_group = {_strip_prefix(x) for x in exclusion_group}
                    if m in canonical_group:
                        others = canonical_group - {m}
                        for x in others:
                            if x in metrics_indexes:
                                col_idx = metrics_indexes[x]
                                excluded_positions.add((row_idx, col_idx))

                for metrics_group in combinations.values():
                    canonical_group = [_strip_prefix(x) for x in metrics_group]
                    if m in canonical_group:
                        others = set(canonical_group) - {m}
                        for y in others:
                            if y in metrics_indexes:
                                col_idx = metrics_indexes[y]
                                excluded_positions.add((row_idx, col_idx))

        data_array = data.to_numpy()

        if by_metric_exclusions:
            exclusions_map = {
                _strip_prefix(metric): [_strip_prefix(x) for x in excluded]
                for metric, excluded in exclusions.items()
            }

            for row_idx in range(data_array.shape[0]):
                considered_metrics = set()

                while True:
                    min_value = np.inf
                    min_col_idx = -1

                    for col_idx, metric_value in enumerate(data_array[row_idx]):
                        if col_idx not in considered_metrics and (row_idx, col_idx) not in excluded_positions:
                            if metric_value < min_value:
                                min_value = metric_value
                                min_col_idx = col_idx

                    if min_col_idx == -1:
                        break

                    considered_metrics.add(min_col_idx)

                    min_metric_label = data.columns[min_col_idx]
                    min_metric_name = _strip_prefix(min_metric_label)
                    excluded_metrics = exclusions_map.get(min_metric_name, [])

                    for excluded_metric in excluded_metrics:
                        if excluded_metric in metrics_indexes:
                            excluded_col_idx = metrics_indexes[excluded_metric]
                            if (row_idx, excluded_col_idx) not in excluded_positions:
                                excluded_positions.add((row_idx, excluded_col_idx))
                                data_array[row_idx, excluded_col_idx] = np.inf

        for new_metric_name, metrics_to_combine in combinations.items():
            canonical_metrics = [_strip_prefix(m) for m in metrics_to_combine]
            c_indexes = [metrics_indexes[m] for m in canonical_metrics if m in metrics_indexes]

            if c_indexes:
                combined_min = np.min(data_array[:, c_indexes], axis=1)

                if np.all(np.isinf(combined_min)):
                    print(f"Skipping combination for '{new_metric_name}' due to incompatible exclusions.")
                    continue
                self.rosetta_data["metric_" + new_metric_name] = combined_min
            else:
                raise ValueError(f"No valid metrics to combine for '{new_metric_name}'.")

        if drop:
            self.rosetta_data.drop(columns=all_metrics_columns, inplace=True)

        for new_metric_name, metrics_to_combine in combinations.items():
            non_excluded_found = False
            canonical_metrics = [_strip_prefix(m) for m in metrics_to_combine]

            for metric in canonical_metrics:
                col_idx = metrics_indexes.get(metric)
                if col_idx is not None:
                    column_values = data_array[:, col_idx]
                    if not np.all(np.isinf(column_values)):
                        non_excluded_found = True
                        break

            if not non_excluded_found:
                print(f"Warning: No non-excluded metrics available to combine for '{new_metric_name}'.")

    def getBestRosettaModels(
        self, filter_values, n_models=1, return_failed=False, exclude_models=None
    ):
        """
        Get best rosetta models based on their best "total_score" and a set of metrics
        with specified thresholds. The filter thresholds must be provided with a dictionary
        using the metric names as keys and the thresholds as the values.

        Parameters
        ==========
        n_models : int
            The number of models to select for each protein + ligand docking.
        filter_values : dict
            Thresholds for the filter.
        return_failed : bool
            Whether to return a list of the dockings without any models fulfilling
            the selection criteria. It is returned as a tuple (index 0) alongside
            the filtered data frame (index 1).
        exclude_models : list
            List of models to be excluded from the best poses selection.
        """

        if exclude_models == None:
            exclude_models = []

        best_poses = pd.DataFrame()
        bp = []
        failed = []
        for model in self.rosetta_data.index.levels[0]:

            if model in exclude_models:
                continue

            model_data = self.rosetta_data[
                self.rosetta_data.index.get_level_values("Model") == model
            ]
            for metric in filter_values:
                if not metric.startswith("metric_"):
                    metric_label = "metric_" + metric
                else:
                    metric_label = metric
                model_data = model_data[
                    model_data[metric_label] < filter_values[metric]
                ]
                if model_data.empty:
                    if model not in failed:
                        failed.append(model)
                    continue
                if model_data.shape[0] < n_models:
                    print(
                        "WARNING: less than %s models passed the filter %s + %s"
                        % (n_models, model, ligand)
                    )
                for i in model_data["score"].nsmallest(n_models).index:
                    bp.append(i)

        if return_failed:
            return failed, self.rosetta_data[self.rosetta_data.index.isin(bp)]
        return self.rosetta_data[self.rosetta_data.index.isin(bp)]

    def getBestRosettaModelsIteratively(
        self,
        metrics,
        n_models=1,
        distance_step=0.1,
        angle_step=1.0,
        fixed=None,
        max_distance=None,
        max_distance_step_shift=None,
        verbose=False,
        rosetta_df=None,
    ):
        """
        Iteratively select the best Rosetta poses per model by progressively relaxing metric thresholds.

        This mirrors :meth:`getBestRosettaDockingPosesIteratively`, but operates on
        ``self.rosetta_data`` (or a user-provided DataFrame) and reuses
        :meth:`getBestRosettaModels` to pick the best-scoring poses as soon as each model
        satisfies the current thresholds. After every iteration the metric that is filtering
        out the most unsatisfied models is relaxed by ``distance_step`` or ``angle_step``.

        Parameters
        ----------
        metrics : dict
            Mapping of metric name -> threshold (scalar upper limit or (lower, upper) range).
        n_models : int, optional
            Number of poses to retain per model when a filter passes. Default 1.
        distance_step : float, optional
            Increment applied to distance metrics when relaxing thresholds.
        angle_step : float, optional
            Increment applied to angle metrics when relaxing thresholds.
        fixed : sequence or str, optional
            Metrics that must not be relaxed.
        max_distance : float, optional
            Cap distance metrics at this value. When reached, optionally switch to ``max_distance_step_shift``.
        max_distance_step_shift : float, optional
            Distance step size to adopt once ``max_distance`` is reached.
        verbose : bool, optional
            Print progress messages during the relaxation loop.
        rosetta_df : pandas.DataFrame, optional
            DataFrame indexed by (Model, Pose) to operate on. Defaults to ``self.rosetta_data``.
        """

        data = rosetta_df if rosetta_df is not None else self.rosetta_data
        if data is None or data.empty:
            raise ValueError("No Rosetta model data available. Run analyseRosettaCalculation first.")

        if fixed is None:
            fixed = []
        elif isinstance(fixed, str):
            fixed = [fixed]

        metrics = metrics.copy()
        non_fixed_metrics = set(metrics.keys()) - set(fixed)
        if not non_fixed_metrics:
            raise ValueError("You must leave at least one metric not fixed")

        # Ensure metric types are known
        for metric in metrics.keys():
            metric_label = metric if metric.startswith("metric_") else "metric_" + metric
            metric_type = self._get_rosetta_metric_type(metric_label)
            if metric_type is None:
                inferred_type = None
                if metric_label in data.columns:
                    if "angle" in metric_label.lower() or "torsion" in metric_label.lower():
                        inferred_type = "angle"
                    else:
                        inferred_type = "distance"
                if inferred_type is not None:
                    self._set_rosetta_metric_type(metric_label, inferred_type)
                else:
                    raise ValueError(
                        f"Metric type for {metric_label} not defined. "
                        "Ensure metrics were registered via combineRosettaDistancesIntoMetric or "
                        "_set_rosetta_metric_type."
                    )

        models = list(data.index.get_level_values(0).unique())
        satisfied_models: Set[str] = set()
        selected_indexes = []
        current_distance_step = distance_step
        step_shift_applied = False

        while len(satisfied_models) < len(models):
            if verbose:
                ti = time.time()

            best_poses = self.getBestRosettaModels(
                metrics, n_models=n_models, exclude_models=list(satisfied_models)
            )

            newly_selected = set()
            for idx in best_poses.index:
                model = idx[0]
                if model not in satisfied_models:
                    selected_indexes.append(idx)
                    newly_selected.add(model)

            satisfied_models.update(newly_selected)

            if len(satisfied_models) >= len(models):
                break

            remaining_models = set(models) - satisfied_models
            remaining_mask = data.index.get_level_values(0).isin(remaining_models)
            remaining_data = data[remaining_mask]

            if remaining_data.empty:
                if verbose:
                    print("No remaining data to relax thresholds on. Terminating iteration.")
                break

            metric_acceptance = {}
            for metric in metrics:
                if metric in fixed:
                    continue
                metric_label = metric if metric.startswith("metric_") else "metric_" + metric
                metric_type = self._get_rosetta_metric_type(metric_label)
                if metric_type is None:
                    raise ValueError(f"Metric type for {metric_label} not defined.")
                metric_values = remaining_data[metric_label]
                threshold = metrics[metric]

                if isinstance(threshold, (int, float)):
                    if metric_type in ["distance", "angle"]:
                        acceptance = metric_values <= threshold
                    else:
                        acceptance = metric_values >= threshold
                elif isinstance(threshold, (tuple, list)):
                    lower, upper = threshold
                    acceptance = (metric_values >= lower) & (metric_values <= upper)
                else:
                    raise ValueError(f"Invalid threshold type for metric {metric}")

                metric_acceptance[metric] = acceptance.sum()

            ordered_metrics = sorted(
                [(m, a) for m, a in metric_acceptance.items() if m not in fixed],
                key=lambda x: x[1],
            )

            updated = False
            for metric, _ in ordered_metrics:
                metric_label = metric if metric.startswith("metric_") else "metric_" + metric
                metric_type = self._get_rosetta_metric_type(metric_label)
                if metric_type == "distance":
                    step = current_distance_step
                elif metric_type == "angle":
                    step = angle_step
                else:
                    raise ValueError(f"Unknown metric type for {metric_label}")

                threshold = metrics[metric]
                if isinstance(threshold, (int, float)):
                    new_value = threshold + step

                    if metric_type == "distance" and max_distance is not None:
                        if not step_shift_applied and new_value >= max_distance:
                            if max_distance_step_shift is not None:
                                current_distance_step = max_distance_step_shift
                                step_shift_applied = True
                                if verbose:
                                    print(
                                        f"Max distance {max_distance} reached for metric {metric}. "
                                        f"Applying step shift to {current_distance_step}."
                                    )
                            else:
                                new_value = max_distance
                                metrics[metric] = new_value
                                if verbose:
                                    print(
                                        f"Max distance {max_distance} reached for metric {metric}. "
                                        "Terminating iteration."
                                    )
                                updated = True
                                break

                    metrics[metric] = new_value
                    updated = True
                    break

                elif isinstance(threshold, (tuple, list)):
                    lower, upper = threshold
                    metrics[metric] = (lower - step, upper + step)
                    updated = True
                    break
                else:
                    raise ValueError(f"Invalid threshold type for metric {metric}")

            if not updated:
                if verbose:
                    print("No metrics were updated. Terminating iteration.")
                break

            if verbose:
                elapsed_time = time.time() - ti
                print(
                    f"Selected models: {len(satisfied_models)}/{len(models)}, "
                    f"Current thresholds: {metrics}, Time elapsed: {elapsed_time:.2f}s",
                    end="\r",
                )

        if selected_indexes:
            return data.loc[selected_indexes]
        return pd.DataFrame()

    def rosettaFilterByProtonationStates(self, residue_states=None, inplace=False):
        """
        Filter the rosetta_data attribute based on the fufillment of protonation state conditions. Protonations states
        defintions must be given through the residue_states attribute. The input is a dictionary with model names as keys
        and as values lists of tuples with the following format: [((chain_id, residue_id), protonation_state), etc.]

        The function is currently implemented to only work with histidine residues.

        Parameters
        ==========
        residue_states : dict
            By model and residue definition of protonation states.
        inplace : bool
            Overwrites the self.rosetta_data by the filtered data frame.

        Returns
        =======
        filtered_data : pandas.DataFrame
            self.rosetta_data dataframe filterd by protonation states.
        """

        data = self.rosetta_protonation_states.reset_index()
        data.columns = [c.replace(" ", "_") for c in data.columns]

        filtered_models = []
        filtered_rows = []

        old_model = None
        histidines = []
        for index, row in data.iterrows():
            ti = time.time()
            model_tag = row.description

            # Empty hisitidine list
            if model_tag != old_model:

                # Check protonation states are in data
                keep_model = True
                if histidines != []:
                    model_base_name = "_".join(model_tag.split("_")[:-1])
                    for rs in residue_states[model_base_name]:
                        if rs not in histidines:
                            keep_model = False

                # Store model
                if keep_model and histidines != []:
                    filtered_models.append(model_tag)

                histidines = []

            histidines.append(((row.chain, row.residue), (row.residue_state)))

            # Update current model as old
            old_model = model_tag

        # filter the rosetta_data attribute
        mask = []
        rosetta_data = self.rosetta_data.reset_index()
        for index, row in rosetta_data.iterrows():
            if row.description in filtered_models:
                mask.append(True)
            else:
                mask.append(False)

        filtered_data = self.rosetta_data[mask]
        if inplace:
            self.rosetta_data = filtered_data

        return filtered_data

    def loadMutantsAsNewModels(
        self,
        mutants_folder,
        filter_score_term="score",
        tags=None,
        min_value=True,
        wat_to_hoh=True,
        keep_model_name=True,
        only_mutants=None,
        cst_files=None,
    ):
        """
        Load the best energy models from a set of silent files inside a createMutants()
        calculation folder. The models are added to the list of models and do not replace
        any previous model already present in the library.

        Parameters
        ==========
        mutants_folder : str
            Path to folder where the Mutants output files are contained (see createMutants() function)
        filter_score_term : str
            Score term used to filter models
        tags : dict
            Tags to extract specific models from the mutant optimization
        """

        executable = "extract_pdbs.linuxgccrelease"
        models = []

        if only_mutants == None:
            only_mutants = []

        if isinstance(only_mutants, str):
            only_mutants = [only_mutants]

        # Check if params were given
        params = None
        if os.path.exists(mutants_folder + "/params"):
            params = mutants_folder + "/params"

        for d in os.listdir(mutants_folder + "/output_models"):
            if os.path.isdir(mutants_folder + "/output_models/" + d):
                for f in os.listdir(mutants_folder + "/output_models/" + d):
                    if f.endswith(".out"):

                        model = d
                        mutant = f.replace(model + "_", "").replace(".out", "")

                        # Read only given mutants
                        if only_mutants != []:
                            if (
                                mutant not in only_mutants
                                and model + "_" + mutant not in only_mutants
                            ):
                                continue

                        scores = readSilentScores(
                            mutants_folder + "/output_models/" + d + "/" + f
                        )
                        if tags != None and mutant in tags:
                            print(
                                "Reading mutant model %s from the given tag %s"
                                % (mutant, tags[mutant])
                            )
                            best_model_tag = tags[mutant]
                        elif min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += (
                            " -silent "
                            + mutants_folder
                            + "/output_models/"
                            + d
                            + "/"
                            + f
                        )
                        if params != None:
                            command += " -extra_res_path " + params + " "
                        command += " -tags " + best_model_tag
                        os.system(command)

                        # Load mutants to the class
                        if keep_model_name:
                            mutant = model + "_" + mutant

                        # self.models_names.append(mutant)
                        self.readModelFromPDB(
                            mutant, best_model_tag + ".pdb", wat_to_hoh=wat_to_hoh
                        )
                        os.remove(best_model_tag + ".pdb")
                        models.append(mutant)

        self.getModelsSequences()
        print("Added the following mutants from folder %s:" % mutants_folder)
        print("\t" + ", ".join(models))

    def loadModelsFromRosettaOptimization(
        self,
        optimization_folder,
        filter_score_term="score",
        min_value=True,
        tags=None,
        wat_to_hoh=True,
        return_missing=False,
        sugars=False,
        conect_update=False,
        output_folder=None,
        rosetta_df=None,
        covalent_check=False,
    ):
        """
        Load the best-scoring poses from a Rosetta relax run and either add them to
        the current library or dump them to an ``output_folder``.

        By default the method scans every ``<model>_relax.out`` file under
        ``optimization_folder/output_models/<model>`` and picks the pose with the
        lowest value of ``filter_score_term`` (set ``min_value=False`` to pick the
        highest). When ``rosetta_df`` is supplied the search is restricted to the
        rows present in that dataframe, allowing the user to pre-filter
        ``self.rosetta_data`` (e.g., by metric thresholds) and then feed the subset
        back into this loader. The same ``filter_score_term`` column in
        ``rosetta_df`` is used to choose the minimum/maximum pose per model before
        mapping back to the silent-file tags.

        If ``output_folder`` is provided the extracted pose is saved there after
        removing the pose index from the filename; otherwise the pose is loaded
        into ``self.structures`` directly. Passing ``return_missing=True`` makes the
        function return the set of models that could not be retrieved.

        Parameters
        ==========
        optimization_folder : str
            Path to the folder containing ``output_models/<model>`` subdirectories
            with the relax silent files.
        filter_score_term : str
            Name of the score column (from the silent file or ``rosetta_df``) used
            to decide which pose to keep.
        min_value : bool
            Select the minimum column value when True; select the maximum when
            False.
        tags : dict or None
            Optional mapping ``model -> silent_tag`` to force extraction of a
            specific pose. When present, the tags take precedence over all other
            selection mechanisms.
        wat_to_hoh : bool
            Convert water residue names from WAT to HOH when loading into the
            class.
        return_missing : bool
            When True, return the set of models that were not found/extracted.
        sugars : bool
            Enable Rosetta sugar extraction flags.
        conect_update : bool
            Update CONECT records in the resulting PDBs.
        output_folder : str or None
            Destination folder to place the extracted PDBs instead of loading
            them into the class. The pose index is stripped from the filename.
        rosetta_df : pandas.DataFrame, optional
            DataFrame indexed by (Model, Pose) that limits the poses considered
            for extraction. Typically a filtered subset of ``self.rosetta_data``;
            the same ``filter_score_term`` column is minimized/maximized within
            this dataframe before resolving the corresponding silent tag.
        covalent_check : bool, optional
            Whether to run the covalent-ligand detection that re-sorts residues
            by index after loading (defaults to False to preserve ligand order).
        """

        def getConectLines(pdb_file, format_for_prepwizard=True):

            ace_names = ['CO', 'OP1', 'CP2', '1HP2', '2HP2', '3HP2']

            # Read PDB file
            atom_tuples = {}
            add_one = False
            previous_chain = None
            with open(pdb_file, "r") as f:
                for l in f:
                    if l.startswith("ATOM") or l.startswith("HETATM"):
                        index, name, resname, chain, resid = (
                            int(l[6:11]),        # Atom index
                            l[12:16].strip(),    # Atom name
                            l[17:20].strip(),    # Residue name
                            l[21],               # Chain identifier
                            int(l[22:26]),       # Residue index
                        )

                        if not previous_chain:
                            previous_chain = chain

                        if name in ace_names:
                            resid -= 1

                            if format_for_prepwizard:
                                if name == 'CP2':
                                    name = 'CH3'
                                elif name == 'CO':
                                    name = 'C'
                                elif name == 'OP1':
                                    name = 'O'
                                elif name == '1HP2':
                                    name = '1H'
                                elif name == '2HP2':
                                    name = '2H'
                                elif name == '3HP2':
                                    name = '3H'

                        if resname == 'NMA':
                            add_one = True

                            if format_for_prepwizard:
                                if name == 'HN2':
                                    name = 'H'
                                elif name == 'C':
                                    name = 'CA'
                                elif name == 'H1':
                                    name = '1HA'
                                elif name == 'H2':
                                    name = '2HA'
                                elif name == 'H3':
                                    name = '3HA'

                        if previous_chain != chain:
                            add_one = False

                        if add_one:
                            resid += 1

                        atom_tuples[index] = (chain, resid, name)
                        previous_chain = chain

            conects = []
            with open(pdb_file) as pdbf:
                for l in pdbf:
                    if l.startswith("CONECT"):
                        l = l.replace("CONECT", "")
                        l = l.strip("\n").rstrip()
                        num = len(l) / 5
                        new_l = [int(l[i * 5 : (i * 5) + 5]) for i in range(int(num))]
                        conects.append([atom_tuples[int(x)] for x in new_l])

            return conects

        def writeConectLines(conects, pdb_file):

            atom_indexes = {}
            with open(pdb_file, "r") as f:
                for l in f:
                    if l.startswith("ATOM") or l.startswith("HETATM"):
                        index, name, resname, chain, resid = (
                            int(l[6:11]),        # Atom index
                            l[12:16].strip(),    # Atom name
                            l[17:20].strip(),    # Residue name
                            l[21],               # Chain identifier
                            int(l[22:26]),       # Residue index
                        )
                        atom_indexes[(chain, resid, name)] = index

            # Check atoms not found in conects
            with open(pdb_file + ".tmp", "w") as tmp:
                with open(pdb_file) as pdb:
                    # write all lines but skip END line
                    for line in pdb:
                        if not line.startswith("END"):
                            tmp.write(line)

                    # Write new conect line mapping
                    for entry in conects:
                        line = "CONECT"
                        for x in entry:
                            line += "%5s" % atom_indexes[x]
                        line += "\n"
                        tmp.write(line)
                tmp.write("END\n")
            shutil.move(pdb_file + ".tmp", pdb_file)

        def checkCappingGroups(pdb_file, format_for_prepwizard=True, keep_conects=True):

            ace_names = ['CO', 'OP1', 'CP2', '1HP2', '2HP2', '3HP2']

            if keep_conects:
                conect_lines = getConectLines(pdb_file)

            # Detect capping groups
            structure = _readPDB(pdb_file, best_model_tag+".pdb")
            model = structure[0]

            for chain in model:

                add_one = False
                residues = [r for r in chain]

                # Check for ACE atoms
                ace_atoms = []
                for a in residues[0]:
                    if a.name in ace_names:
                        ace_atoms.append(a)

                # Check for NMA residue
                nma_residue = None
                for r in residues:
                    if r.resname == 'NMA':
                        nma_residue = r

                # Build a separate residue for ACE
                new_chain = PDB.Chain.Chain(chain.id)

                if ace_atoms:

                    for a in ace_atoms:
                        residues[0].detach_child(a.name)

                    ace_residue = PDB.Residue.Residue((' ', residues[0].id[1]-1, ' '), 'ACE', '')

                    for i, a in enumerate(ace_atoms):
                        new_name = a.get_name()

                        # Define the new name based on the old one
                        if format_for_prepwizard:
                            if new_name == 'CP2':
                                new_name = 'CH3'
                            elif new_name == 'CO':
                                new_name = 'C'
                            elif new_name == 'OP1':
                                new_name = 'O'
                            elif new_name == '1HP2':
                                new_name = '1H'
                            elif new_name == '2HP2':
                                new_name = '2H'
                            elif new_name == '3HP2':
                                new_name = '3H'

                        # Create a new atom
                        new_atom = PDB.Atom.Atom(
                            new_name,                  # Atom name
                            a.get_coord(),             # Coordinates
                            a.get_bfactor(),           # B-factor
                            a.get_occupancy(),         # Occupancy
                            a.get_altloc(),            # AltLoc
                            "%-4s" % new_name,         # Full atom name (formatted)
                            a.get_serial_number(),     # Serial number
                            a.element                  # Element symbol
                        )

                        ace_residue.add(new_atom)

                    new_chain.add(ace_residue)

                # Renumber residues and rename atoms
                for i, r in enumerate(residues):

                    # Handle NMA residue atom renaming
                    if r == nma_residue and format_for_prepwizard:
                        renamed_atoms = []
                        for a in nma_residue:

                            new_name = a.get_name()  # Original atom name

                            # Rename the atom based on the rules
                            if new_name == 'HN2':
                                new_name = 'H'
                            elif new_name == 'C':
                                new_name = 'CA'
                            elif new_name == 'H1':
                                new_name = '1HA'
                            elif new_name == 'H2':
                                new_name = '2HA'
                            elif new_name == 'H3':
                                new_name = '3HA'

                            # Create a new atom with the updated name
                            new_atom = PDB.Atom.Atom(
                                new_name,                  # New name
                                a.get_coord(),             # Same coordinates
                                a.get_bfactor(),           # Same B-factor
                                a.get_occupancy(),         # Same occupancy
                                a.get_altloc(),            # Same altloc
                                "%-4s" % new_name,         # Full atom name (formatted)
                                a.get_serial_number(),     # Same serial number
                                a.element                  # Same element
                            )
                            renamed_atoms.append(new_atom)

                        # Create a new residue with renamed atoms
                        nma_residue = PDB.Residue.Residue(r.id, r.resname, r.segid)
                        for atom in renamed_atoms:
                            nma_residue.add(atom)

                        r = nma_residue
                        add_one = True

                    if add_one:
                        chain.detach_child(r.id)  # Deatach residue from old chain
                        new_id = (r.id[0], r.id[1]+1, r.id[2])  # New ID with updated residue number
                        r.id = new_id  # Update residue ID with renumbered value

                    # Add residue to the new chain
                    new_chain.add(r)

                model.detach_child(chain.id)
                model.add(new_chain)

            _saveStructureToPDB(structure, pdb_file)

            if keep_conects:
                writeConectLines(conect_lines, pdb_file)

        def _build_pose_tag_lookup(score_index):
            lookup = {}
            for tag in score_index:
                suffix = tag.split("_")[-1]
                lookup[suffix] = tag
                try:
                    lookup[int(suffix)] = tag
                except ValueError:
                    pass
            return lookup

        rosetta_subset = None
        rosetta_subset_models: Optional[Set[str]] = None
        if rosetta_df is not None:
            if not isinstance(rosetta_df, pd.DataFrame) or rosetta_df.empty:
                raise ValueError(
                    "rosetta_df must be a non-empty pandas.DataFrame indexed by (Model, Pose)."
                )
            if filter_score_term not in rosetta_df.columns:
                raise ValueError(
                    f"Column '{filter_score_term}' not found in provided rosetta_df."
                )
            if not isinstance(rosetta_df.index, pd.MultiIndex):
                raise ValueError(
                    "rosetta_df index must be a MultiIndex with 'Model' and 'Pose' levels."
                )
            index_names = list(rosetta_df.index.names)
            if "Model" not in index_names or "Pose" not in index_names:
                raise ValueError(
                    "rosetta_df index must include 'Model' and 'Pose' levels."
                )
            rosetta_subset = rosetta_df
            rosetta_subset_models = set(
                rosetta_subset.index.get_level_values("Model")
            )

        executable = "extract_pdbs.linuxgccrelease"
        models = []

        # Check if params were given
        params = None
        if os.path.exists(optimization_folder + "/params"):
            params = optimization_folder + "/params"
            patch_line = ""
            for p in os.listdir(params):
                if not p.endswith(".params"):
                    patch_line += params + "/" + p + " "

        if output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        for d in os.listdir(optimization_folder + "/output_models"):
            subfolder = os.path.join(optimization_folder, "output_models", d)
            if os.path.isdir(subfolder):
                for f in os.listdir(subfolder):
                    if f.endswith("_relax.out"):
                        model = d

                        # Skip models not loaded into the library
                        if model not in self.models_names:
                            continue

                        if rosetta_subset_models is not None and model not in rosetta_subset_models:
                            continue

                        scores = readSilentScores(os.path.join(subfolder, f))
                        if tags is not None and model in tags:
                            print("Reading model %s from the given tag %s" % (model, tags[model]))
                            best_model_tag = tags[model]
                        elif rosetta_subset is not None:
                            try:
                                model_subset = rosetta_subset.xs(model, level="Model")
                            except KeyError:
                                continue

                            if model_subset.empty:
                                continue

                            score_series = model_subset[filter_score_term].dropna()
                            if score_series.empty:
                                continue

                            pose_identifier = (
                                score_series.idxmin()
                                if min_value
                                else score_series.idxmax()
                            )
                            if isinstance(pose_identifier, tuple):
                                pose_key = pose_identifier[-1]
                            else:
                                pose_key = pose_identifier

                            pose_lookup = _build_pose_tag_lookup(scores.index)
                            best_model_tag = pose_lookup.get(pose_key)
                            if best_model_tag is None:
                                pose_str = str(pose_key)
                                best_model_tag = pose_lookup.get(pose_str)
                            if best_model_tag is None:
                                raise ValueError(
                                    f"Pose {pose_key} for model {model} was not found in {f}."
                                )
                        elif min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += " -silent " + os.path.join(subfolder, f)
                        if params is not None:
                            command += " -extra_res_path " + params
                            if patch_line != "":
                                command += " -extra_patch_fa " + patch_line
                        command += " -tags " + best_model_tag
                        if sugars:
                            command += " -include_sugars"
                            command += " -alternate_3_letter_codes pdb_sugar"
                            command += " -write_glycan_pdb_codes"
                            command += " -auto_detect_glycan_connections"
                            command += " -maintain_links"
                        os.system(command)

                        checkCappingGroups(best_model_tag + ".pdb", keep_conects=False)

                        # Remove the pose index from the name.
                        base_name = '_'.join(best_model_tag.split("_")[:-1])
                        new_filename = base_name + ".pdb"
                        if output_folder:
                            os.rename(best_model_tag + ".pdb", os.path.join(output_folder, new_filename))
                        else:
                            self.readModelFromPDB(
                                model,
                                best_model_tag + ".pdb",
                                wat_to_hoh=wat_to_hoh,
                                conect_update=conect_update,
                                covalent_check=covalent_check,
                            )
                            os.remove(best_model_tag + ".pdb")
                        models.append(model)

        self.getModelsSequences()

        expected_models = set(self.models_names)
        if rosetta_subset_models is not None:
            expected_models &= rosetta_subset_models

        missing_models = expected_models - set(models)
        if missing_models:
            print("Missing models in relaxation folder:")
            print("\t" + ", ".join(missing_models))
            if return_missing:
                return missing_models

    def loadModelsFromMissingLoopBuilding(
        self, job_folder, filter_score_term="score", min_value=True, param_files=None
    ):
        """
        Load models from addMissingLoops() job calculation output.

        Parameters:
        job_folder : str
            Path to the addMissingLoops() calculation folder containing output.
        """

        # Get silent models paths
        executable = "extract_pdbs.linuxgccrelease"
        output_folder = job_folder + "/output_models"
        models = []

        # Check if params were given
        params = None
        if os.path.exists(job_folder + "/params"):
            params = job_folder + "/params"

        # Check loops to rebuild from output folder structure
        for model in os.listdir(output_folder):
            model_folder = output_folder + "/" + model
            loop_models = {}
            for loop in os.listdir(model_folder):
                loop_folder = model_folder + "/" + loop
                for f in os.listdir(loop_folder):
                    # If rebuilded loops are found get best structures.
                    if f.endswith(".out"):
                        scores = readSilentScores(loop_folder + "/" + f)
                        best_model_tag = scores.idxmin()[filter_score_term]
                        if min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += " -silent " + loop_folder + "/" + f
                        if params != None:
                            command += " -extra_res_path " + params + " "
                        command += " -tags " + best_model_tag
                        os.system(command)
                        loop = (int(loop.split("_")[0]), loop.split("_")[1])
                        loop_models[loop] = _readPDB(loop, best_model_tag + ".pdb")
                        os.remove(best_model_tag + ".pdb")
                        models.append(model)

            if len(loop_models) > 1:
                # Get original model chains
                model_chains = [*self.structures[model].get_chains()]

                # Create new structure, model and chains to add rebuilded segments
                structure = PDB.Structure.Structure(0)
                _model = PDB.Model.Model(0)
                chains = {}
                for model_chain in model_chains:
                    chains[model_chain.id] = PDB.Chain.Chain(model_chain.id)

                # Add missing loop segments to overall model
                current_residue = 0

                for loop in loop_models:
                    # Add loop remodel protocol
                    if len(loop[1]) == 1:
                        hanging_residues = 3
                    elif len(loop[1]) == 2:
                        hanging_residues = 2
                    else:
                        hanging_residues = 1
                    larger_loop_residue = loop[0] + len(loop[1]) + 1 + hanging_residues
                    for i, residue in enumerate(loop_models[loop].get_residues()):
                        if i + 1 > current_residue and i + 1 <= larger_loop_residue:
                            chain_id = residue.get_parent().id
                            chains[chain_id].add(residue)
                            current_residue += 1

                # Load final model into the library
                for chain in chains:
                    _model.add(chains[chain])
                structure.add(_model)
                _saveStructureToPDB(structure, model + ".pdb")
            else:
                for loop in loop_models:
                    _saveStructureToPDB(loop_models[loop], model + ".pdb")

            self.readModelFromPDB(model, model + ".pdb")
            os.remove(model + ".pdb")

        missing_models = set(self.models_names) - set(models)
        if missing_models != set():
            print("Missing models in loop rebuild folder:")
            print("\t" + ", ".join(missing_models))

    def loadModelsFromMembranePositioning(self, job_folder):
        """ """
        for model in os.listdir(job_folder + "/output_models"):
            pdb_path = job_folder + "/output_models/" + model + "/" + model + ".pdb"
            self.readModelFromPDB(model, pdb_path)

    def setUpRFDiffusion(
        self,
        job_folder,
        only_models=None,
        num_designs=100,
        num_batches=1,
        diffuser_T=None,
        contig=None,              # e.g., 'A1-218/30-30'
        partial_T=None,           # e.g., 10
        provide_seq=None,         # e.g., [218, 247]
        script_path=None,
        additional_args=None,      # dict of other overrides
        gpu_local=False
    ):
        """
        Create shell commands for RFdiffusion jobs, splitting each model into N batches.
        You’ll get len(models) * num_batches commands in `jobs`.

        Parameters:
        - self.models_paths: dict model_name -> path_to_pdb
        - job_folder (str): Base folder for inputs/outputs
        - num_designs (int): Number of designs per batch
        - num_batches (int): How many independent batches per model
        - contig (str): contigmap.contigs value, e.g. 'A1-218/30-30'
        - partial_T (int): diffuser.partial_T value
        - provide_seq (list[int] or str): contigmap.provide_seq value
        - script_path (str): path to run_inference.py
        - additional_args (dict): any other key: value RFdiffusion overrides
        - gpu_local (bool): If True, sets the CUDA_VISIBLE_DEVICES variable for
                            running GPUs in local computer with multiple GPUS.

        Returns:
        - jobs (list[str]): one shell-command string per batch per model
        """

        # ensure folders exist
        os.makedirs(job_folder, exist_ok=True)
        input_folder = os.path.join(job_folder, 'input_models')
        os.makedirs(input_folder, exist_ok=True)
        output_folder = os.path.join(job_folder, 'output_models')
        os.makedirs(output_folder, exist_ok=True)

        if not script_path:
            script_path = 'SCRIPT_PATH/run_inference.py'

        # Convert contig into a per model dictionary
        if isinstance(contig, str):
            contig_dict = {}
            for model in self:
                contig_dict[model] = contig
            contig = contig_dict

        jobs = []
        for model, pdb_path in self.models_paths.items():

            if only_models and model not in only_models:
                continue

            # copy once per model
            shutil.copyfile(pdb_path, os.path.join(input_folder, f"{model}.pdb"))

            # create num_batches commands for this model
            for batch_id in range(num_batches):

                batch_folder = os.path.join(output_folder, f"batch_{batch_id}")
                os.makedirs(batch_folder, exist_ok=True)

                # include batch in output_prefix
                output_prefix = f"designs/{model}"

                cmd = []
                cmd.append('cd '+batch_folder+';')
                if gpu_local:
                    cmd.append('CUDA_VISIBLE_DEVICES=GPUID')
                cmd.append(script_path)
                cmd.append(f"inference.input_pdb=../../input_models/{model}.pdb")
                cmd.append(f"inference.output_prefix={output_prefix}")
                cmd.append(f"inference.num_designs={num_designs}")
                if diffuser_T:
                    cmd.append(f"diffuser.T={diffuser_T}")
                if contig:
                    cmd.append(f"contigmap.contigs=[{contig[model]}]")
                if partial_T is not None:
                    cmd.append(f"diffuser.partial_T={partial_T}")
                if provide_seq is not None:
                    if isinstance(provide_seq, (list, tuple)) and len(provide_seq) == 2:
                        cmd.append(f"contigmap.provide_seq=[{provide_seq[0]}-{provide_seq[1]}]")
                    else:
                        cmd.append(f"contigmap.provide_seq={provide_seq}")

                if additional_args:
                    for key, value in additional_args.items():
                        if isinstance(value, bool):
                            value = str(value).lower()
                        cmd.append(f"{key}={value}")
                cmd.append('; cd ../../../')

                # join with backslashes for readability
                jobs.append(" \\\n  ".join(cmd))

        return jobs

    def setUpBoltz2Calculation(
        self,
        job_folder,
        diffusion_samples=1,
        use_msa_server=True,
        msa_path=None,
        sampling_steps=200,
        recycling_steps=3,
        output_format="pdb",
        use_potentials=False,
        ligands=None,
        binder=None,
        chains=None
    ):
        """
        Run Boltz2 structure prediction and afinity.

        It can be use to predict ligand binding with multiple ligands.
        Additionally, it can be used to predict ligand binding affinity, but for just 1 ligand
        Ligand/s should be given in SMILEs format.

        To run in MN5 you need to precompute the MSA with AF3 or some other method!!!

        Parameters:
        - diffusion_samples: int. The number of diffusion samples to use for prediction.
        - use_msa_server: Bool. Whether to use the MSA server for sampling, it auto-generate the MSA using the mmseqs2 server (not for MN5) because no internet.
        - msa_path: str. Path to the folder with the .a3m files.  It should contain the pre-computed MSA for the proteins. You need to run AF3 first (or something else) to generate the MSA
            The name of the .a3m files should be the name of the model.
            To run single sequence mode input "empty" for msa_path.
            !!Not tested!!
        - sampling_steps: int. The number of sampling steps to use for prediction.
        - recycling_steps: int. The number of recycling steps to use for prediction.
            AF3 uses by default 25 diffusion_samples and 10 recycling_steps.
        - use_potentials: bool. Whether to use the potentials for prediction. Set to True should improve performance, but uses more memmory.
        - ligands (list[str]): list of ligands SMILEs
        - binder (str): chain of the binder/ligand to use to predict affinity. Only 1 binder is allowed

        chains : None, str, iterable or dict, optional
            Chain selector per model used to pick the sequence for the Boltz2
            input. Defaults to the first polymer chain per model.

        Returns:
        - jobs (list[str]): command string per batch per model
        """

        # Need to write yaml, and then need to write jobs
        os.makedirs(job_folder, exist_ok=True)

        for model in self.models_names:
            if not os.path.exists(job_folder + "/" +model):
                os.mkdir(job_folder+"/"+model)

        chain_selection = self._resolve_chain_selection(
            chains, require_single=True
        )

        for model in self.models_names:
            chain_id = chain_selection.get(model)
            sequence = None

            if chain_id is not None:
                sequence = self.chain_sequences.get(model, {}).get(chain_id)
            else:
                sequence = self.sequences.get(model)

            if sequence in (None, ""):
                raise ValueError(
                    f"No protein sequence found for model {model} (chain {chain_id})."
                )

        # Write the YAML file
            with open(job_folder+"/"+model +"/" + "boltz.yaml", "w") as iyf:
                iyf.write('version: 1\n')
                iyf.write('sequences:\n')
                iyf.write('  - protein:\n')
                iyf.write(f'      id: [{chain_id}]\n')
                iyf.write(f'      sequence: {sequence}\n')

                if use_msa_server == False:
                    iyf.write(f'      msa: {msa_path}+{model}.a3m\n')

                if ligands != None:
                    for i, ligand in enumerate(ligands):
                        ligand_chain = chr(ord('B') + i)  # Assign chains starting from 'B'
                        iyf.write('  - ligand:\n')
                        iyf.write(f'      id: [{ligand_chain}]\n')
                        iyf.write(f"      smiles: '{ligand}'\n")

                if binder != None:
                    iyf.write('properties:\n')
                    iyf.write('    - affinity:\n')
                    iyf.write(f'        binder: {binder}\n')


        jobs = []
        for model in self.models_names:
            pdb_name = f"{model}.pdb"
            pdb_path = job_folder+"/"+model

            command = f"cd {pdb_path} \n"
            command +=  "boltz predict boltz.yaml "

            if use_msa_server == True:
                command += "--use_msa_server "
            else:
                if msa_path is None:
                    raise ValueError("msa_path must be provided if use_msa_server is False")
            if use_potentials:
                command += "--use_potentials "
            if diffusion_samples:
                command += f"--diffusion_samples {diffusion_samples} "
            if recycling_steps:
                command += f"--recycling_steps {recycling_steps} "
            if sampling_steps:
                command += f"--sampling_steps {sampling_steps} "
            if output_format:
                command += f"--output_format {output_format} "

            jobs.append(command)

        return jobs


    def saveModels(
        self,
        output_folder,
        keep_residues={},
        models=None,
        convert_to_mae=False,
        convert_to_mol2=False,
        write_conect_lines=True,
        replace_symbol=None,
        add_cryst1_record=False,
        **keywords,
    ):
        """
        Save all models as PDBs into the output_folder.

        Parameters
        ==========
        output_folder : str
            Path to the output folder to store models.
        add_cryst1_record : bool, optional
            When True, a standard CRYST1 line is prepended if missing.
        """
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if convert_to_mae:
            _copyScriptFile(output_folder, "PDBtoMAE.py")
            script_name = "._PDBtoMAE.py"
        if convert_to_mol2:
            _copyScriptFile(output_folder, "PDBtoMOL2.py")
            script_name = "._PDBtoMOL2.py"

        if replace_symbol:
            if not isinstance(replace_symbol, tuple) or len(replace_symbol) != 2:
                raise ValueError(
                    "replace_symbol must be a tuple (old_symbol, new_symbol)"
                )

        for model in self.models_names:

            if replace_symbol:
                model_name = model.replace(replace_symbol[0], replace_symbol[1])
            else:
                model_name = model

            pdb_output_path = os.path.join(output_folder, f"{model_name}.pdb")

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            # Get residues to keep
            if model in keep_residues:
                kr = keep_residues[model]
            else:
                kr = []

            _saveStructureToPDB(
                self.structures[model],
                pdb_output_path,
                keep_residues=kr,
                **keywords,
            )

            if add_cryst1_record:
                _ensure_cryst1_record(pdb_output_path)

            if "remove_hydrogens" in keywords:
                if keywords["remove_hydrogens"] == True:
                    check_file = True
                    hydrogens = False
                else:
                    check_file = False
                    hydrogens = True
            else:
                check_file = False
                hydrogens = True

            if write_conect_lines:
                self._write_conect_lines(
                    model,
                    pdb_output_path,
                    check_file=check_file,
                    hydrogens=hydrogens,
                )

        if convert_to_mae:
            command = "cd " + output_folder + "\n"
            command += "run ._PDBtoMAE.py\n"
            command += "cd ../\n"
            os.system(command)
        if convert_to_mol2:
            command = "cd " + output_folder + "\n"
            command += "run ._PDBtoMOL2.py\n"
            command += "cd ../\n"
            os.system(command)

    def removeModel(self, model):
        """
        Removes a specific model from this class

        Parameters
        ==========
        model : str
            Model name to remove
        """
        missing = []
        try:
            self.models_paths.pop(model)
        except KeyError:
            missing.append("models_paths")
        try:
            self.models_names.remove(model)
        except ValueError:
            missing.append("models_names")
        try:
            self.structures.pop(model)
        except KeyError:
            missing.append("structures")
        try:
            self.sequences.pop(model)
        except KeyError:
            missing.append("sequences")
        if missing:
            raise ValueError(f"Model {model!r} not present in: {', '.join(missing)}")

    def readTargetSequences(self, fasta_file):
        """
        Read the set of target sequences for the protein models
        """
        # Read sequences and store them in target_sequence attributes
        sequences = prepare_proteins.alignment.readFastaFile(fasta_file)
        for sequence in sequences:
            if sequence not in self.models_names:
                print(
                    "Given sequence name %s does not matches any protein model"
                    % sequence
                )
            else:
                self.target_sequences[sequence] = sequences[sequence]

        missing_models = set(self.models_names) - set(self.target_sequences)
        if missing_models != set():
            print("Missing sequences in the given fasta file:")
            print("\t" + ", ".join(missing_models))

    def compareSequences(self, sequences_file, chain=None):
        """
        Compare models sequences to their given sequences and check for missing
        or changed sequence information.

        Parameters
        ==========
        sequences_file : str
            Path to the fasta file containing the sequences to compare. The model
            names must match.

        Returns
        =======
        sequence_differences : dict
            Dictionary containing missing or changed information.
        """

        if self.multi_chain and chain == None:
            raise ValueError("PDBs contain multiple chains. Please select one chain.")

        self.readTargetSequences(sequences_file)

        # Iterate models to store sequence differences
        for model in self.models_names:

            if model not in self.target_sequences:
                message = (
                    "Sequence for model %s not found in the given fasta file! " % model
                )
                message += "Please make sure to include one sequence for each model "
                message += "loaded into prepare proteins."
                raise ValueError(message)

            # Create lists for missing information
            self.sequence_differences[model] = {}
            self.sequence_differences[model]["n_terminus"] = []
            self.sequence_differences[model]["mutations"] = []
            self.sequence_differences[model]["missing_loops"] = []
            self.sequence_differences[model]["c_terminus"] = []

            # Create a sequence alignement between current and target sequence
            to_align = {}
            if chain:
                to_align["current"] = self.sequences[model][chain]
            else:
                to_align["current"] = self.sequences[model]
            to_align["target"] = self.target_sequences[model]
            msa = prepare_proteins.alignment.mafft.multipleSequenceAlignment(
                to_align, stderr=False, stdout=False
            )

            # Iterate the alignment to gather sequence differences
            p = 0
            n = True
            loop_sequence = ""
            loop_start = 0

            # Check for n-terminus, mutations and missing loops
            for i in range(msa.get_alignment_length()):
                csp = msa[0].seq[i]
                tsp = msa[1].seq[i]
                if csp != "-":
                    p += 1
                if csp == "-" and tsp != "-" and n:
                    self.sequence_differences[model]["n_terminus"].append(tsp)
                elif csp != "-" and tsp != "-":
                    n = False
                    if (
                        loop_sequence != "" and len(loop_sequence) > 1
                    ):  # Ignore single-residue loops
                        self.sequence_differences[model]["missing_loops"].append(
                            (loop_start, loop_sequence)
                        )
                    loop_sequence = ""
                    loop_start = 0

                    if csp != tsp:
                        self.sequence_differences[model]["mutations"].append((p, tsp))

                elif csp == "-" and tsp != "-" and p < len(to_align["current"]):
                    if loop_start == 0:
                        loop_start = p
                    loop_sequence += tsp

            # Check for c-terminus
            for i in reversed(range(msa.get_alignment_length())):
                csp = msa[0].seq[i]
                tsp = msa[1].seq[i]
                if csp == "-" and tsp != "-":
                    self.sequence_differences[model]["c_terminus"].append(tsp)
                elif csp != "-" and tsp != "-":
                    break

            self.sequence_differences[model]["n_terminus"] = "".join(
                self.sequence_differences[model]["n_terminus"]
            )
            self.sequence_differences[model]["c_terminus"] = "".join(
                reversed(self.sequence_differences[model]["c_terminus"])
            )

        return self.sequence_differences

    def _write_conect_lines(
        self, model, pdb_file, atom_mapping=None, check_file=False, hydrogens=True
    ):
        """
        Write stored conect lines for a particular model into the given PDB file.

        Parameters
        ==========
        model : str
            Model name
        pdb_file : str
            Path to PDB file to modify
        """

        def check_atom_in_atoms(atom, atoms, atom_mapping):
            if atom_mapping != None:
                atom_mapping = atom_mapping[model]

            if atom not in atoms and atom_mapping != None and atom in atom_mapping:
                if isinstance(atom_mapping[atom], str):
                    atom = (atom[0], atom[1], atom_mapping[atom])
                elif (
                    isinstance(atom_mapping[atom], tuple)
                    and len(atom_mapping[atom]) == 3
                ):
                    atom = atom_mapping[atom]

            if atom not in atoms:
                residue_atoms = " ".join([ac[-1] for ac in atoms if atom[1] == ac[1]])
                message = "Conect atom %s not found in %s's topology\n\n" % (
                    atom,
                    pdb_file,
                )
                message += "Topology's residue %s atom names: %s" % (
                    atom[1],
                    residue_atoms,
                )
                raise ValueError(message)

            return atom

        # Get atom indexes map
        atoms = self._getAtomIndexes(
            model, pdb_file, invert=True, check_file=check_file
        )

        # Check atoms not found in conects
        with open(pdb_file + ".tmp", "w") as tmp:
            with open(pdb_file) as pdb:

                # write all lines but skip END line
                for line in pdb:
                    if not line.startswith("END"):
                        tmp.write(line)

                # Write new conect line mapping
                for entry in self.conects[model]:
                    line = "CONECT"
                    for x in entry:
                        if not hydrogens:
                            type_index = x[2].find(next(filter(str.isalpha, x[2])))
                            if x[2][type_index] != "H":
                                x = check_atom_in_atoms(
                                    x, atoms, atom_mapping=atom_mapping
                                )
                                line += "%5s" % atoms[x]
                        else:
                            x = check_atom_in_atoms(x, atoms, atom_mapping=atom_mapping)
                            line += "%5s" % atoms[x]

                    line += "\n"
                    tmp.write(line)
            tmp.write("END\n")
        shutil.move(pdb_file + ".tmp", pdb_file)

    def _getChainSequence(self, chain):
        """
        Get the one-letter protein sequence of a Bio.PDB.Chain object.

        Parameters
        ----------
        chain : Bio.PDB.Chain
            Input chain to retrieve its sequence from.

        Returns
        -------
        sequence : str
            Sequence of the input protein chain.
        None
            If chain does not contain protein residues.
        """
        sequence = ""
        for r in chain:

            filter = False
            if r.resname in ["HIE", "HID", "HIP"]:
                resname = "HIS"
            elif r.resname == "CYX":
                resname = "CYS"
            elif r.resname == "ASH":
                resname = "ASP"
            elif r.resname == "GLH":
                resname = "GLU"
            else:
                # Leave out HETATMs
                if r.id[0] != " ":
                    filter = True
                resname = r.resname

            if not filter:  # Non heteroatom filter
                try:
                    sequence += _three_to_one(resname)
                except:
                    sequence += "X"

        if sequence == "":
            return None
        else:
            return sequence

    def _checkCovalentLigands(
        self, model, pdb_file, atom_mapping=None, check_file=False
    ):
        """ """
        self.covalent[model] = []  # Store covalent residues
        if check_file:
            structure = _readPDB(model, pdb_file)
        else:
            structure = self.structures[model]

        # Iterate chains in model structure
        for c in structure[0]:

            indexes = []  # Store residue indexes
            hetero = []  # Store heteroatom residue indexes
            residues = []  # Store residues orderly (for later)
            for r in c:
                indexes.append(r.id[1])
                if r.id[0].startswith("H_"):
                    hetero.append(r.id[1])
                residues.append(r)

            # Check for individual and other gaps
            gaps2 = []  # Store individual gaps
            other_gaps = []  # Store other gaps
            for i in range(len(indexes)):
                if i > 0:
                    if indexes[i] - indexes[i - 1] == 2:
                        gaps2.append((indexes[i - 1], indexes[i]))
                    elif indexes[i] - indexes[i - 1] != 1:
                        other_gaps.append(indexes[i])

            # Check if individual gaps can be filled with any residue in other_gaps
            for g2 in gaps2:
                for og in other_gaps:
                    if g2[1] - og == 1 and og - g2[0] == 1:

                        if check_file:
                            print(
                                "Found misplaced residue %s for file %s"
                                % (og, pdb_file)
                            )
                        else:
                            print(
                                "Found misplaced residue %s for model %s" % (og, model)
                            )

                        print("Possibly a covalent-link exists for this HETATM residue")
                        print(
                            "Sorting residues by their indexes... to disable pass covalent_check=False."
                        )

                        self._sortStructureResidues(
                            model,
                            pdb_file,
                            check_file=check_file,
                            atom_mapping=atom_mapping,
                        )
                        self.covalent[model].append(og)

            # Check if hetero-residue is found between two non-hetero residues
            for i, r in enumerate(residues):
                if r.id[1] in hetero and r.resname not in ["HIP", "HID", "HIE"]:
                    if i + 1 == len(residues):
                        continue
                    chain = r.get_parent()
                    pr = residues[i - 1]
                    nr = residues[i + 1]
                    if (
                        pr.get_parent().id == chain.id
                        and nr.get_parent().id == chain.id
                    ):
                        if pr.id[0] == " " and nr.id[0] == " ":
                            self.covalent[model].append(r.id[1])

    def _sortStructureResidues(
        self, model, pdb_file, atom_mapping=None, check_file=False
    ):

        # Create new structure
        n_structure = PDB.Structure.Structure(0)

        # Create new model
        n_model = PDB.Model.Model(self.structures[model][0].id)

        if check_file:
            structure = _readPDB(model, pdb_file)
        else:
            structure = self.structures[model]

        # Iterate chains from old model
        model = [m for m in structure][0]
        for chain in model:
            n_chain = PDB.Chain.Chain(chain.id)

            # Gather residues
            residues = []
            for r in chain:
                residues.append(r)

            # Iterate residues orderly by their ID
            for r in sorted(residues, key=lambda x: x.id[1]):
                n_chain.add(r)

            n_model.add(n_chain)
        n_structure.add(n_model)

        _saveStructureToPDB(n_structure, pdb_file + ".tmp")
        self._write_conect_lines(
            model, pdb_file + ".tmp", atom_mapping=atom_mapping, check_file=check_file
        )
        shutil.move(pdb_file + ".tmp", pdb_file)
        n_structure = _readPDB(model, pdb_file)

        # Update structure model in library
        if not check_file:
            self.structures[model] = n_structure

    def _readPDBConectLines(self, pdb_file, model, only_hetatoms=False):
        """
        Read PDB file and get conect lines only
        """

        # Get atom indexes by tuple and objects
        atoms = self._getAtomIndexes(model, pdb_file)
        if only_hetatoms:
            atoms_objects = self._getAtomIndexes(model, pdb_file, return_objects=True)

        conects = []
        # Read conect lines as dictionaries linking atoms
        with open(pdb_file) as pdbf:
            for l in pdbf:
                if l.startswith("CONECT"):
                    l = l.replace("CONECT", "")
                    l = l.strip("\n").rstrip()
                    num = len(l) / 5
                    new_l = [int(l[i * 5 : (i * 5) + 5]) for i in range(int(num))]
                    if only_hetatoms:
                        het_atoms = [
                            (
                                True
                                if atoms_objects[int(x)].get_parent().id[0] != " "
                                else False
                            )
                            for x in new_l
                        ]
                        if True not in het_atoms:
                            continue
                    conects.append([atoms[int(x)] for x in new_l])
        return conects

    def _getAtomIndexes(
        self, model, pdb_file, invert=False, check_file=False, return_objects=False
    ):

        # Read PDB file
        atom_indexes = {}
        with open(pdb_file, "r") as f:
            for l in f:
                if l.startswith("ATOM") or l.startswith("HETATM"):
                    index, name, chain, resid = (
                        int(l[6:11]),
                        l[12:16].strip(),
                        l[21],
                        int(l[22:26]),
                    )
                    atom_indexes[(chain, resid, name)] = index

        if check_file:
            structure = _readPDB(model, pdb_file)
        else:
            structure = self.structures[model]

        # Assign PDB indexes to each Bio.PDB atom
        atoms = {}
        for chain in structure[0]:
            for residue in chain:
                for atom in residue:

                    # Get atom PDB index
                    index = atom_indexes[(chain.id, residue.id[1], atom.name)]

                    # Return atom objects instead of tuples
                    if return_objects:
                        _atom = atom
                    else:
                        _atom = _get_atom_tuple(atom)

                    # Invert the returned dictionary
                    if invert:
                        atoms[_atom] = index
                    else:
                        atoms[index] = _atom
        return atoms

    def _maybe_prepare_maestro_models(self, models_folder, maestro_export_options):
        """
        When the provided models_folder is a Maestro file, invoke the export script
        to generate per-pose PDB files and return the path to the temporary folder.
        """
        path = Path(models_folder).expanduser()
        if not path.is_file():
            return models_folder
        lowered = path.name.lower()
        if not lowered.endswith((".mae", ".maegz")):
            return models_folder

        output_dir = tempfile.mkdtemp(prefix="maestro_models_")
        manifest_path = os.path.join(output_dir, "maestro_manifest.json")
        script_path = pkg_resources.resource_filename(
            "prepare_proteins.scripts", "export_maestro_models.py"
        )

        schrodinger_root = os.environ.get("SCHRODINGER")
        schrodinger_run = None
        if schrodinger_root:
            candidate = os.path.join(schrodinger_root, "run")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                schrodinger_run = candidate

        run_executable = shutil.which("run")
        python_cmd = "python3" if shutil.which("python3") else "python"

        if run_executable:
            launcher = [run_executable, python_cmd]
            launcher_name = run_executable
        elif schrodinger_run:
            launcher = [schrodinger_run, python_cmd]
            launcher_name = schrodinger_run
        else:
            launcher = [sys.executable]
            launcher_name = sys.executable

        cmd = launcher + [
            script_path,
            str(path),
            "--output-dir",
            output_dir,
            "--manifest",
            manifest_path,
        ]

        option_map = {
            "prefix": "--prefix",
            "ligand_chain": "--ligand-chain",
            "separator": "--separator",
        }
        int_option_map = {
            "protein_ct": "--protein-ct",
            "ligand_resnum": "--ligand-resnum",
        }
        bool_option_map = {"keep_original_ligand_ids": "--keep-original-ligand-ids"}
        allowed_keys = set(option_map) | set(int_option_map) | set(bool_option_map)

        if maestro_export_options:
            for key, value in maestro_export_options.items():
                if key not in allowed_keys:
                    raise ValueError(
                        f"Unsupported maestro_export_options key '{key}'. "
                        "Valid keys are: prefix, protein_ct, ligand_chain, "
                        "ligand_resnum, keep_original_ligand_ids, separator."
                    )
                if key in option_map and value is not None:
                    cmd.extend([option_map[key], str(value)])
                elif key in int_option_map and value is not None:
                    cmd.extend([int_option_map[key], str(int(value))])
                elif key in bool_option_map and value:
                    cmd.append(bool_option_map[key])

        used_run_launcher = launcher_name != sys.executable

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            stdout = exc.stdout or ""
            extra = ""
            if "No module named 'schrodinger'" in stderr:
                extra = (
                    " The invoked Python interpreter cannot import the 'schrodinger' "
                    "module. Make sure the Schrodinger 'run' command is available "
                    "(it is typically added to PATH after sourcing the Schrodinger "
                    "environment) or set the SCHRODINGER environment variable so the "
                    "launcher can be located automatically."
                )
            elif not used_run_launcher:
                extra = (
                    " Neither the 'run' command nor the SCHRODINGER environment "
                    "variable were detected. The current Python interpreter was used "
                    "instead, which typically cannot import the 'schrodinger' module. "
                    "Ensure the Schrodinger environment is sourced so that 'run' is "
                    "available on PATH."
                )
            raise RuntimeError(
                "Failed to convert Maestro file into PDB models."
                f"{extra}\nCommand: {' '.join(cmd)}\nStdout:\n{stdout}\nStderr:\n{stderr}"
            ) from exc

        pdb_files = [
            entry
            for entry in os.listdir(output_dir)
            if entry.lower().endswith(".pdb")
        ]
        if not pdb_files:
            raise RuntimeError(
                f"No PDB files were produced from Maestro file {path}. "
                "Check that the file contains at least one ligand pose."
            )

        manifest_data = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as mf:
                    manifest_data = json.load(mf)
            except json.JSONDecodeError:
                manifest_data = None

        self._maestro_temp_dir = output_dir
        self._maestro_manifest = manifest_data
        self._maestro_manifest_path = manifest_path if os.path.exists(manifest_path) else None
        self._maestro_source_file = str(path)

        return output_dir

    def _getModelsPaths(self, only_models=None, exclude_models=None):
        """
        Get PDB models paths in the models_folder attribute

        Returns
        =======

        paths : dict
            Paths to all models
        """

        paths = {}
        for d in os.listdir(self.models_folder):
            if d.endswith(".pdb"):
                pdb_name = ".".join(d.split(".")[:-1])

                if only_models != []:
                    if pdb_name not in only_models:
                        continue

                if exclude_models != []:
                    if pdb_name in exclude_models:
                        continue

                paths[pdb_name] = self.models_folder + "/" + d

        return paths

    def __iter__(self):
        # returning __iter__ object
        self._iter_n = -1
        self._stop_inter = len(self.models_names)
        return self

    def __next__(self):
        self._iter_n += 1
        if self._iter_n < self._stop_inter:
            return self.models_names[self._iter_n]
        else:
            raise StopIteration


def readSilentScores(silent_file):
    """
    Read scores from a silent file into a Pandas DataFrame object.

    Parameters
    ==========
    silent_file : str
        Path to the silent file.

    Returns
    =======
    scores : Pandas.DataFrame
        Rosetta score for each model.
    """

    scores = {}
    terms = []
    with open(silent_file) as sf:
        for l in sf:
            if l.startswith("SCORE"):
                if terms == []:
                    terms = l.strip().split()
                    for t in terms:
                        scores[t] = []
                else:
                    for i, t in enumerate(terms):
                        try:
                            if '_' in l.strip().split()[i]:
                                scores[t].append(l.strip().split()[i])
                            else:
                                scores[t].append(float(l.strip().split()[i]))
                        except:
                            scores[t].append(l.strip().split()[i])
    scores = pd.DataFrame(scores)
    scores.pop("SCORE:")
    scores = pd.DataFrame(scores)
    scores = scores.set_index("description")
    scores = scores.sort_index()

    return scores


def _readPDB(name, pdb_file):
    """
    Read PDB file to a structure object
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(name, pdb_file)
    return structure


# Note: parallel loading was explored but removed as it did not
# provide speedups in practice for typical workloads and added
# complexity. Serial loading plus reduced I/O proved more robust.

def _saveStructureToPDB(
    structure,
    output_file,
    remove_hydrogens=False,
    remove_water=False,
    only_protein=False,
    keep_residues=[],
):
    """
    Saves a structure into a PDB file

    Parameters
    ----------
    structure : list or Bio.PDB.Structure
        Structure to save
    remove_hydrogens : bool
        Remove hydrogen atoms from model?
    remove_water : bool
        Remove water residues from model?
    only_protein : bool
        Remove everything but the protein atoms?
    keep_residues : list
        List of residue indexes to keep when using the only_protein selector.
    """

    io = PDB.PDBIO()
    io.set_structure(structure)

    selector = None
    if remove_hydrogens:
        selector = _atom_selectors.notHydrogen()
    elif remove_water:
        selector = _atom_selectors.notWater()
    elif only_protein:
        selector = _atom_selectors.onlyProtein(keep_residues=keep_residues)
    if selector != None:
        io.save(output_file, selector)
    else:
        io.save(output_file)

_DEFAULT_CRYST1_RECORD = "CRYST1   1.000   1.000   1.000  90.00  90.00  90.00 P 1           1"

def _ensure_cryst1_record(pdb_path, cryst1_record=_DEFAULT_CRYST1_RECORD):
    """
    Make sure the CRYST1 record exists at the top of the PDB file.
    """
    record = cryst1_record.rstrip("\n") + "\n"
    with open(pdb_path, "r") as existing:
        lines = existing.readlines()

    if lines and lines[0].startswith("CRYST1"):
        return

    with open(pdb_path, "w") as updated:
        updated.write(record)
        updated.writelines(lines)

def _copyScriptFile(
    output_folder, script_name, no_py=False, subfolder=None, hidden=True, path="prepare_proteins/scripts",
):
    """
    Copy a script file from the prepare_proteins package.

    Parameters
    ==========

    """
    # Get script

    if subfolder != None:
        path = path + "/" + subfolder

    script_file = resource_stream(
        Requirement.parse("prepare_proteins"), path + "/" + script_name
    )
    script_file = io.TextIOWrapper(script_file)

    # Write control script to output folder
    if no_py == True:
        script_name = script_name.replace(".py", "")

    if hidden:
        output_path = output_folder + "/._" + script_name
    else:
        output_path = output_folder + "/" + script_name

    with open(output_path, "w") as sof:
        for l in script_file:
            sof.write(l)


def make_model_path_batches(
    models,
    batch_size=None,
    n_batches=None,
    only_models=None,
    exclude_models=None,
    shuffle=False,
    seed=None,
):
    """
    Create batches of model paths without loading structures.

    Parameters
    ==========
    models : str | dict | list
        - str: folder containing PDB files
        - dict: mapping {model_name: pdb_path}
        - list/tuple: list of PDB file paths
    batch_size : int, optional
        Number of models per batch. If None, `n_batches` must be provided.
    n_batches : int, optional
        Number of batches to split into. Ignored if `batch_size` is provided.
    only_models : list|tuple|set|str, optional
        Subset of model names to include.
    exclude_models : list|tuple|set|str, optional
        Subset of model names to exclude.
    shuffle : bool, default False
        Randomize model order before batching.
    seed : int, optional
        Random seed when `shuffle=True`.

    Returns
    =======
    dict
        Mapping of batch_index -> {model_name: pdb_path}
    """

    # Collect paths {name: path}
    if isinstance(models, str):
        if not os.path.isdir(models):
            raise ValueError("models path must be a directory when a string is provided")
        paths = {}
        for d in os.listdir(models):
            if d.endswith(".pdb"):
                name = ".".join(d.split(".")[:-1])
                paths[name] = os.path.join(models, d)
    elif isinstance(models, dict):
        paths = dict(models)
    elif isinstance(models, (list, tuple)):
        paths = {}
        for p in models:
            if not isinstance(p, str) or not p.endswith(".pdb"):
                raise ValueError("All items in models list must be PDB file paths")
            name = os.path.basename(p)
            name = ".".join(name.split(".")[:-1])
            paths[name] = p
    else:
        raise ValueError("models must be a folder path, dict, or list of PDB paths")

    # Filters
    if only_models is None:
        only_models = []
    elif isinstance(only_models, str):
        only_models = [only_models]

    if exclude_models is None:
        exclude_models = []
    elif isinstance(exclude_models, str):
        exclude_models = [exclude_models]

    names = list(paths.keys())

    # Apply include/exclude
    if only_models:
        names = [n for n in names if n in set(only_models)]
    if exclude_models:
        names = [n for n in names if n not in set(exclude_models)]

    # Order or shuffle
    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(names)
    else:
        names = sorted(names)

    total = len(names)
    if total == 0:
        return {}

    # Determine batching
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        n_batches_calc = (total + batch_size - 1) // batch_size
    else:
        if not n_batches or n_batches <= 0:
            raise ValueError("Provide batch_size or a positive n_batches")
        n_batches_calc = n_batches
        batch_size = (total + n_batches_calc - 1) // n_batches_calc

    # Build batches
    batches = {}
    for i in range(n_batches_calc):
        start = i * batch_size
        end = min(start + batch_size, total)
        if start >= end:
            break
        batch_names = names[start:end]
        batches[i] = {n: paths[n] for n in batch_names}

    return batches

def _computeCartesianFromInternal(coord2, coord3, coord4, distance, angle, torsion):
    """
    Compute the cartesian coordinates for the i atom based on internal coordinates
    of other three atoms (j, k, l).

    Parameters
    ==========
    coord1 : numpy.ndarray shape=(3,)
        Coordinate of the j atom bound to the i atom
    coord2 : numpy.ndarray shape=(3,)
        Coordinate of the k atom bound to the j atom
    coord3 : numpy.ndarray shape=(3,)
        Coordinate of the l atom bound to the k atom
    distance : float
        Distance between the i and j atoms in angstroms
    angle : float
        Angle between the i, j, and k atoms in degrees
    torsion : float
        Torsion between the i, j, k, l atoms in degrees

    Returns
    =======
    coord1 : float
        Coordinate of the i atom

    """

    torsion = torsion * np.pi / 180.0  # Convert to radians
    angle = angle * np.pi / 180.0  # Convert to radians

    v1 = coord2 - coord3
    v2 = coord2 - coord4

    n = np.cross(v1, v2)
    nn = np.cross(v1, n)

    n /= np.linalg.norm(n)
    nn /= np.linalg.norm(nn)

    n *= -np.sin(torsion)
    nn *= np.cos(torsion)

    v3 = n + nn
    v3 /= np.linalg.norm(v3)
    v3 *= distance * np.sin(angle)

    v1 /= np.linalg.norm(v1)
    v1 *= distance * np.cos(angle)

    coord1 = coord2 + v3 - v1

    return coord1


def _get_atom_tuple(atom):
    return (atom.get_parent().get_parent().id, atom.get_parent().id[1], atom.name)


def _getStructureCoordinates(
    structure,
    as_dict=False,
    return_atoms=False,
    only_protein=False,
    sidechain=False,
    backbone=False,
    only_residues=None,
    exclude_residues=None,
):
    """
    Get the coordinates for each atom in the structure.
    """

    if as_dict:
        if return_atoms:
            raise ValueError("as_dict and return_atoms are not compatible!")
        coordinates = {}
    else:
        coordinates = []

    for atom in structure.get_atoms():
        residue = atom.get_parent()
        chain = residue.get_parent()
        residue_tuple = (chain.id, residue.id[1])
        atom_tuple = (chain.id, residue.id[1], atom.name)

        if exclude_residues and residue_tuple in exclude_residues:
            continue

        if only_residues and residue_tuple not in only_residues:
            continue

        if only_protein or sidechain or backbone:
            if residue.id[0] != " ":
                continue

        if sidechain:
            if atom.name in ["N", "CA", "C", "O"]:
                continue

        elif backbone:
            if atom.name not in ["N", "CA", "C", "O"]:
                continue

        if as_dict:
            coordinates[atom_tuple] = atom.coord
        elif return_atoms:
            coordinates.append(atom_tuple)
        else:
            coordinates.append(atom.coord)

    if not as_dict:
        coordinates = np.array(coordinates)

    return coordinates

def _readRosettaScoreFile(score_file, indexing=False, skip_empty=False):
    """
    Reads a Rosetta score file and returns a DataFrame of the scores.

    Arguments:
    ==========
    score_file : str
        Path to the input score file.
    indexing : bool, optional
        If True, sets the DataFrame index to ['Model', 'Pose'].

    Returns:
    ========
    DataFrame
        A DataFrame containing the scores from the score file.
    """
    with open(score_file) as sf:
        lines = [x.strip() for x in sf if x.startswith("SCORE:")]

    if len(lines) < 2:
        if not skip_empty:
            raise ValueError("The score file does not contain enough data.")
        else:
            return None

    score_terms = lines[0].split()[1:]  # Get the terms excluding the initial "SCORE:"
    scores = {term: [] for term in score_terms}
    models = []
    poses = []
    descriptions = []

    for line in lines[1:]:
        parts = line.split()[1:]  # Get the parts excluding the initial "SCORE:"
        if len(parts) != len(score_terms):
            continue  # Skip lines that are headers or do not match the number of score terms
        if parts[0] == score_terms[0]:  # Check if this is a repeated header
            continue

        for i, score in enumerate(score_terms):
            try:
                scores[score].append(float(parts[i]))
            except ValueError:
                scores[score].append(parts[i])

        # Extract model and pose from the 'description' field
        description_index = score_terms.index("description")
        description = parts[description_index]
        model, pose = "_".join(description.split("_")[:-1]), description.split("_")[-1]
        models.append(model)
        poses.append(int(pose))
        descriptions.append(description)

    scores.pop("description")
    scores["Model"] = np.array(models)
    scores["Pose"] = np.array(poses)
    scores["description"] = np.array(descriptions)

    scores_df = pd.DataFrame(scores)

    if indexing:
        scores_df = scores_df.set_index(["Model", "Pose"])

    return scores_df
