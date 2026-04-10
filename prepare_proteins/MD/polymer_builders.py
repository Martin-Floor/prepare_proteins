"""Reusable specifications for polymer system construction.

This module provides a small, dependency-light API for describing polymer
systems before committing to a specific coordinate-generation backend. The
initial focus is on building validated specifications for representative
polymer chains, bundles, and surface fragments that can later be translated
into OpenMM-ready coordinates and topologies.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "PolymerAtomTemplate",
    "PolymerRepeatUnit",
    "PolymerResidueTemplate",
    "PolymerTemplate",
    "CelluloseCrystalResidue",
    "CelluloseCrystalChain",
    "CelluloseCrystalSurfaceResidue",
    "CelluloseCrystalSurface",
    "PETCrystalResidue",
    "PETCrystalChain",
    "PolymerChainSpec",
    "PolymerBuildSpec",
    "PolymerBuilder",
    "assign_pdb_chain_ids_by_ter",
]


def _clean_label(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty.")
    return text


def _normalize_chain_ids(n_chains: int, chain_ids: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    if n_chains < 1:
        raise ValueError("n_chains must be at least 1.")

    if chain_ids is None:
        default_ids = []
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for idx in range(n_chains):
            quotient, remainder = divmod(idx, len(alphabet))
            chain_id = alphabet[remainder] if quotient == 0 else f"{alphabet[quotient - 1]}{alphabet[remainder]}"
            default_ids.append(chain_id)
        return tuple(default_ids)

    normalized = tuple(_clean_label(cid, "chain_id") for cid in chain_ids)
    if len(normalized) != n_chains:
        raise ValueError(f"Expected {n_chains} chain IDs, got {len(normalized)}.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("chain_ids must be unique.")
    return normalized


def _normalize_xyz(values: Sequence[float], label: str) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three coordinates.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _add_xyz(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def _scale_xyz(vec: Sequence[float], scale: float) -> Tuple[float, float, float]:
    factor = float(scale)
    return (float(vec[0]) * factor, float(vec[1]) * factor, float(vec[2]) * factor)


def _guess_element(atom_name: str) -> str:
    name = _clean_label(atom_name, "atom_name").strip()
    letters = "".join(ch for ch in name if ch.isalpha())
    if not letters:
        return "X"
    if len(letters) >= 2 and letters[:2].title() in {"Cl", "Br", "Na", "Mg", "Ca", "Fe", "Zn", "Cu"}:
        return letters[:2].title()
    return letters[0].upper()


def assign_pdb_chain_ids_by_ter(
    input_pdb: Union[str, Path],
    output_pdb: Union[str, Path],
    *,
    chain_ids: Optional[Sequence[str]] = None,
    start_chain_id: str = "A",
) -> str:
    """Rewrite ``input_pdb`` assigning chain IDs segment-by-segment using ``TER`` records.

    This is useful for Amber/tleap-generated carbohydrate PDBs that preserve chain
    separation with ``TER`` but leave the chain-ID column blank.
    """
    input_path = Path(input_pdb).resolve()
    output_path = Path(output_pdb).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if chain_ids is None:
        start = _clean_label(start_chain_id, "start_chain_id")
        if len(start) != 1:
            raise ValueError("start_chain_id must be a single character.")
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        try:
            start_index = alphabet.index(start)
        except ValueError as exc:
            raise ValueError("start_chain_id must be an alphanumeric single character.") from exc
        assigned_chain_ids = alphabet[start_index:]
    else:
        assigned_chain_ids = tuple(_clean_label(cid, "chain_id") for cid in chain_ids)
        if any(len(cid) != 1 for cid in assigned_chain_ids):
            raise ValueError("All chain_ids must be single characters.")

    chain_index = 0
    segment_has_atoms = False
    rewritten = []
    for line in input_path.read_text().splitlines():
        record = line[:6].strip()
        if record in {"ATOM", "HETATM"}:
            if chain_index >= len(assigned_chain_ids):
                raise ValueError(
                    "Not enough chain IDs were provided to cover all TER-separated segments."
                )
            chain_id = assigned_chain_ids[chain_index]
            if len(line) < 22:
                line = line.ljust(22)
            rewritten.append(f"{line[:21]}{chain_id}{line[22:]}")
            segment_has_atoms = True
            continue
        rewritten.append(line)
        if record == "TER" and segment_has_atoms:
            chain_index += 1
            segment_has_atoms = False

    output_path.write_text("\n".join(rewritten) + "\n")
    return str(output_path)


def _normalize_single_char_chain_ids(
    n_chains: int,
    chain_ids: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    if n_chains < 1:
        raise ValueError("n_chains must be at least 1.")

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if chain_ids is None:
        if n_chains > len(alphabet):
            raise ValueError(
                f"PDB output supports at most {len(alphabet)} default single-character chain IDs."
            )
        return tuple(alphabet[idx] for idx in range(n_chains))

    normalized = tuple(_clean_label(cid, "chain_id") for cid in chain_ids)
    if len(normalized) != n_chains:
        raise ValueError(f"Expected {n_chains} chain IDs, got {len(normalized)}.")
    if any(len(cid) != 1 for cid in normalized):
        raise ValueError("All chain_ids must be single characters for PDB output.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("chain_ids must be unique.")
    return normalized


def _format_pdb_atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    position: Sequence[float],
    element: str,
    *,
    record_type: str = "HETATM",
    occupancy: float = 1.0,
    bfactor: float = 0.0,
) -> str:
    padded_atom_name = atom_name.rjust(4) if len(atom_name) < 4 else atom_name[:4]
    return (
        f"{record_type[:6]:<6s}{serial:5d} {padded_atom_name} "
        f"{residue_name[:3]:>3s} {chain_id[:1]:1s}{residue_number:4d}    "
        f"{float(position[0]):8.3f}{float(position[1]):8.3f}{float(position[2]):8.3f}"
        f"{float(occupancy):6.2f}{float(bfactor):6.2f}          {str(element).strip().upper()[:2]:>2s}"
    )


_CELLULOSE_IBETA_A = 7.784
_CELLULOSE_IBETA_B = 8.201
_CELLULOSE_IBETA_C = 10.380
_CELLULOSE_IBETA_GAMMA_DEGREES = 96.5
_CELLULOSE_IBETA_ATOM_NAMES = (
    "C1",
    "H1",
    "C2",
    "H2",
    "C3",
    "H3",
    "C4",
    "H4",
    "C5",
    "H5",
    "C6",
    "H61",
    "H62",
    "O2",
    "O3",
    "O4",
    "O5",
    "O6",
    "HO2",
    "HO3",
    "HO6",
)
_CELLULOSE_IBETA_GLYCAM_ATOM_RENAMES = {"HO2": "H2O", "HO3": "H3O", "HO6": "H6O"}
_CELLULOSE_IBETA_ASYMMETRIC_UNIT = (
    (0.0140, -0.0420, 0.0433),
    (0.1382, -0.0188, 0.0598),
    (-0.0260, -0.1840, -0.0516),
    (-0.1508, -0.2177, -0.0546),
    (0.0400, -0.1370, -0.1848),
    (0.1667, -0.1323, -0.1840),
    (-0.0070, 0.0300, -0.2250),
    (-0.1316, 0.0176, -0.2425),
    (0.0260, 0.1590, -0.1232),
    (0.1507, 0.1821, -0.1090),
    (-0.0470, 0.3190, -0.1530),
    (-0.1672, 0.2959, -0.1778),
    (-0.0407, 0.3872, -0.0765),
    (0.0610, -0.3140, -0.0003),
    (-0.0270, -0.2610, -0.2759),
    (0.0790, 0.0860, -0.3424),
    (-0.0560, 0.0990, -0.0053),
    (0.0480, 0.4030, -0.2540),
    (0.0366, -0.3194, 0.0920),
    (0.0475, -0.2400, -0.3517),
    (-0.0144, 0.4924, -0.2850),
    (0.5330, 0.4550, 0.3043),
    (0.6577, 0.4616, 0.3200),
    (0.4750, 0.3178, 0.2094),
    (0.3480, 0.3058, 0.2052),
    (0.5450, 0.3630, 0.0765),
    (0.6716, 0.3753, 0.0793),
    (0.4820, 0.5260, 0.0375),
    (0.3561, 0.5139, 0.0289),
    (0.5410, 0.6574, 0.1388),
    (0.6670, 0.6830, 0.1370),
    (0.4520, 0.8150, 0.1125),
    (0.3341, 0.7843, 0.0847),
    (0.4491, 0.8778, 0.1916),
    (0.5280, 0.1660, 0.2520),
    (0.4860, 0.2330, -0.0081),
    (0.5630, 0.5860, -0.0806),
    (0.4850, 0.6070, 0.2639),
    (0.5420, 0.9140, 0.0170),
    (0.5035, 0.1593, 0.3448),
    (0.5104, 0.2733, -0.0961),
    (0.5184, 1.0275, 0.0326),
)


def _cellulose_ibeta_fractional_residues() -> Tuple[Tuple[Tuple[float, float, float], ...], ...]:
    image = tuple((-x, -y, z + 0.5) for x, y, z in _CELLULOSE_IBETA_ASYMMETRIC_UNIT)
    full = _CELLULOSE_IBETA_ASYMMETRIC_UNIT + image
    return tuple(tuple(full[offset : offset + 21]) for offset in range(0, len(full), 21))


def _cellulose_ibeta_frac_to_cart(frac: Sequence[float]) -> Tuple[float, float, float]:
    gamma = math.radians(_CELLULOSE_IBETA_GAMMA_DEGREES)
    cosg = math.cos(gamma)
    sing = math.sin(gamma)
    x, y, z = _normalize_xyz(frac, "fractional coordinate")
    return (
        x * _CELLULOSE_IBETA_A + y * _CELLULOSE_IBETA_B * cosg,
        y * _CELLULOSE_IBETA_B * sing,
        z * _CELLULOSE_IBETA_C,
    )


def _default_cellulose_beta14_template() -> PolymerTemplate:
    return PolymerTemplate(
        name="cellulose_beta14_glucose",
        repeat_unit=PolymerRepeatUnit(
            name="glucose_beta14",
            residue_name="GLC",
            head_atom="O4",
            tail_atom="C1",
            atom_names=("C1", "C2", "C3", "C4", "C5", "C6", "O2", "O3", "O4", "O5", "O6"),
        ),
        description="Beta-1,4 glucose repeat suitable for cellulose crystal construction.",
        bond_length_angstrom=1.43,
        metadata={
            "glycam_sugar_code": "G",
            "glycam_anomer": "B",
            "glycam_linkage_position": 4,
            "glycam_reducing_end_cap": "ROH",
        },
    )


def _default_polyethylene_terephthalate_template() -> PolymerTemplate:
    return PolymerTemplate(
        name="polyethylene_terephthalate",
        repeat_unit=PolymerRepeatUnit(
            name="pet_repeat",
            residue_name="PET",
            head_atom="O1",
            tail_atom="C7",
            atom_names=("C1", "O1", "C2", "C3", "O2", "C4", "C5", "C6", "C7"),
        ),
        description="Polyethylene terephthalate repeat for crystal-like chain packing.",
        bond_length_angstrom=1.34,
        metadata={
            "polymer_common_name": "PET",
            "repeat_vector_angstrom": (0.0, 0.0, 10.75),
            "crystal_like_a_spacing_angstrom": 4.56,
            "crystal_like_b_spacing_angstrom": 5.94,
            "crystal_triclinic_a_angstrom": 4.56,
            "crystal_triclinic_b_angstrom": 5.94,
            "crystal_triclinic_c_angstrom": 10.75,
            "crystal_triclinic_alpha_degrees": 98.5,
            "crystal_triclinic_beta_degrees": 118.0,
            "crystal_triclinic_gamma_degrees": 112.0,
        },
    )


_PET_WIMZEX01_ATOM_SITES = (
    ("C1", "C", (0.000, 0.078, 0.629)),
    ("C2", "C", (0.000, 0.181, 0.763)),
    ("C3", "C", (0.900, 0.830, 0.562)),
    ("C4", "C", (0.823, 0.743, 0.433)),
    ("C5", "C", (0.045, 0.915, 0.040)),
    ("O1", "O", (0.902, 0.585, 0.170)),
    ("O2", "O", (0.052, 0.000, 0.175)),
    ("C5A", "C", (0.955, 0.085, 0.960)),
    ("O2A", "O", (0.948, 0.000, 0.825)),
    ("C1A", "C", (0.000, 0.922, 0.371)),
    ("C2A", "C", (0.000, 0.819, 0.237)),
    ("C3A", "C", (0.100, 0.170, 0.438)),
    ("C4A", "C", (0.177, 0.257, 0.567)),
    ("O1A", "O", (0.098, 0.415, 0.830)),
)
_PET_WIMZEX01_ATOM_NAME_ORDER = tuple(site[0] for site in _PET_WIMZEX01_ATOM_SITES)
_PET_WIMZEX01_COVALENT_RADII = {"C": 0.76, "O": 0.66}
_PET_WIMZEX01_BOND_TOLERANCE_ANGSTROM = 0.45


def _frac_to_cart_with_lattice_vectors(
    frac: Sequence[float],
    a_vec: Sequence[float],
    b_vec: Sequence[float],
    c_vec: Sequence[float],
) -> Tuple[float, float, float]:
    x, y, z = _normalize_xyz(frac, "fractional coordinate")
    return _add_xyz(_add_xyz(_scale_xyz(a_vec, x), _scale_xyz(b_vec, y)), _scale_xyz(c_vec, z))


def _positive_neighbor_cell_offsets() -> Tuple[Tuple[int, int, int], ...]:
    offsets = [(0, 0, 0)]
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (da, db, dc) == (0, 0, 0):
                    continue
                if da > 0 or (da == 0 and db > 0) or (da == 0 and db == 0 and dc > 0):
                    offsets.append((da, db, dc))
    return tuple(offsets)


_POSITIVE_NEIGHBOR_CELL_OFFSETS = _positive_neighbor_cell_offsets()


def _triclinic_lattice_vectors_with_c_along_z(
    a_angstrom: float,
    b_angstrom: float,
    c_angstrom: float,
    alpha_degrees: float,
    beta_degrees: float,
    gamma_degrees: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    a_length = float(a_angstrom)
    b_length = float(b_angstrom)
    c_length = float(c_angstrom)
    if a_length <= 0.0 or b_length <= 0.0 or c_length <= 0.0:
        raise ValueError("Triclinic cell lengths must all be positive.")

    alpha = math.radians(float(alpha_degrees))
    beta = math.radians(float(beta_degrees))
    gamma = math.radians(float(gamma_degrees))
    for label, angle in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if angle <= 0.0 or angle >= math.pi:
            raise ValueError(f"{label}_degrees must lie strictly between 0 and 180.")

    sin_beta = math.sin(beta)
    if abs(sin_beta) < 1.0e-8:
        raise ValueError("beta_degrees produces a singular triclinic lattice.")

    a_vec = (
        a_length * sin_beta,
        0.0,
        a_length * math.cos(beta),
    )
    b_x = b_length * (math.cos(gamma) - math.cos(alpha) * math.cos(beta)) / sin_beta
    b_z = b_length * math.cos(alpha)
    b_y_sq = b_length * b_length - b_x * b_x - b_z * b_z
    if b_y_sq < -1.0e-6:
        raise ValueError("alpha_degrees, beta_degrees, and gamma_degrees do not define a valid triclinic cell.")
    b_vec = (
        b_x,
        math.sqrt(max(0.0, b_y_sq)),
        b_z,
    )
    c_vec = (0.0, 0.0, c_length)
    return a_vec, b_vec, c_vec


def _format_pdb_cryst1_line(
    a_angstrom: float,
    b_angstrom: float,
    c_angstrom: float,
    alpha_degrees: float,
    beta_degrees: float,
    gamma_degrees: float,
    *,
    space_group: str = "P 1",
    z_value: int = 1,
) -> str:
    return (
        f"CRYST1{float(a_angstrom):9.3f}{float(b_angstrom):9.3f}{float(c_angstrom):9.3f}"
        f"{float(alpha_degrees):7.2f}{float(beta_degrees):7.2f}{float(gamma_degrees):7.2f} "
        f"{str(space_group)[:11]:<11s}{int(z_value):4d}"
    )


def _load_pdb_residues(pdb_path: Union[str, Path]) -> Tuple[Dict[str, Any], ...]:
    residues: list[Dict[str, Any]] = []
    current_key: Optional[Tuple[str, str, int]] = None
    current: Optional[Dict[str, Any]] = None

    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        residue_name = line[17:20].strip()
        chain_id = line[21].strip() or "_"
        residue_id = int(line[22:26])
        key = (residue_name, chain_id, residue_id)
        if key != current_key:
            current = {
                "residue_name": residue_name,
                "chain_id": chain_id,
                "residue_id": residue_id,
                "atoms": {},
            }
            residues.append(current)
            current_key = key
        assert current is not None
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or _guess_element(atom_name)).strip().title()
        current["atoms"][atom_name] = {
            "position": (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ),
            "element": element,
        }
    return tuple(residues)


def _kabsch_transform(
    reference_positions: Sequence[Sequence[float]],
    target_positions: Sequence[Sequence[float]],
) -> Tuple["Any", "Any"]:
    import numpy as np

    reference = np.asarray(reference_positions, dtype=float)
    target = np.asarray(target_positions, dtype=float)
    if reference.shape != target.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference_positions and target_positions must be matching Nx3 coordinate arrays.")

    reference_centroid = reference.mean(axis=0)
    target_centroid = target.mean(axis=0)
    centered_reference = reference - reference_centroid
    centered_target = target - target_centroid
    covariance = centered_reference.T @ centered_target
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_centroid - reference_centroid @ rotation
    return rotation, translation


def _apply_transform(position: Sequence[float], rotation: "Any", translation: "Any") -> Tuple[float, float, float]:
    import numpy as np

    transformed = np.asarray(position, dtype=float) @ rotation + translation
    return (float(transformed[0]), float(transformed[1]), float(transformed[2]))


@dataclass(frozen=True)
class PolymerAtomTemplate:
    """Single atom entry for a reusable residue template."""

    name: str
    element: str
    position: Tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_label(self.name, "atom name"))
        object.__setattr__(self, "element", _clean_label(self.element, "element"))
        object.__setattr__(self, "position", _normalize_xyz(self.position, "position"))


@dataclass(frozen=True)
class PolymerRepeatUnit:
    """Description of a polymer repeat unit and its connection atoms."""

    name: str
    residue_name: str
    head_atom: str
    tail_atom: str
    atom_names: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_label(self.name, "repeat unit name"))
        object.__setattr__(self, "residue_name", _clean_label(self.residue_name, "residue_name").upper())
        object.__setattr__(self, "head_atom", _clean_label(self.head_atom, "head_atom"))
        object.__setattr__(self, "tail_atom", _clean_label(self.tail_atom, "tail_atom"))
        object.__setattr__(self, "atom_names", tuple(_clean_label(atom, "atom_names entry") for atom in self.atom_names))
        if self.atom_names:
            atom_set = set(self.atom_names)
            if self.head_atom not in atom_set:
                raise ValueError(f"head_atom {self.head_atom!r} is not present in atom_names.")
            if self.tail_atom not in atom_set:
                raise ValueError(f"tail_atom {self.tail_atom!r} is not present in atom_names.")


@dataclass(frozen=True)
class PolymerTemplate:
    """Reusable polymer template built from a repeat unit."""

    name: str
    repeat_unit: PolymerRepeatUnit
    description: str = ""
    bond_length_angstrom: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_label(self.name, "template name"))
        if self.bond_length_angstrom is not None and self.bond_length_angstrom <= 0:
            raise ValueError("bond_length_angstrom must be positive when provided.")

    @property
    def linkage_atoms(self) -> Tuple[str, str]:
        return (self.repeat_unit.tail_atom, self.repeat_unit.head_atom)


@dataclass(frozen=True)
class PolymerResidueTemplate:
    """Coordinate template for a single polymer residue/repeat unit."""

    residue_name: str
    atoms: Tuple[PolymerAtomTemplate, ...]
    bonds: Tuple[Tuple[str, str], ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "residue_name", _clean_label(self.residue_name, "residue_name").upper())
        if not self.atoms:
            raise ValueError("PolymerResidueTemplate requires at least one atom.")
        atom_names = [atom.name for atom in self.atoms]
        if len(set(atom_names)) != len(atom_names):
            raise ValueError("PolymerResidueTemplate atom names must be unique.")
        atom_name_set = set(atom_names)
        normalized_bonds = []
        seen_bonds = set()
        for bond in self.bonds:
            if len(bond) != 2:
                raise ValueError("Each PolymerResidueTemplate bond must contain exactly two atom names.")
            atom_a = _clean_label(bond[0], "bond atom name")
            atom_b = _clean_label(bond[1], "bond atom name")
            if atom_a == atom_b:
                raise ValueError("PolymerResidueTemplate bonds must connect two distinct atoms.")
            if atom_a not in atom_name_set or atom_b not in atom_name_set:
                raise ValueError("PolymerResidueTemplate bonds must reference atoms present in the template.")
            bond_key = tuple(sorted((atom_a, atom_b)))
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)
            normalized_bonds.append((atom_a, atom_b))
        object.__setattr__(self, "bonds", tuple(normalized_bonds))

    def get_atom(self, name: str) -> PolymerAtomTemplate:
        """Return the atom with the given name."""
        atom_name = _clean_label(name, "atom name")
        for atom in self.atoms:
            if atom.name == atom_name:
                return atom
        raise KeyError(f"Atom {atom_name!r} not found in residue template {self.residue_name!r}.")


@dataclass(frozen=True)
class CelluloseCrystalResidue:
    """Single glucose placement within a cellulose crystal chain."""

    basis_index: int
    cell_index: Tuple[int, int, int]
    atom_positions: Dict[str, Tuple[float, float, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.basis_index < 1:
            raise ValueError("basis_index must be at least 1.")
        if len(self.cell_index) != 3:
            raise ValueError("cell_index must contain exactly three integers.")
        normalized_positions = {
            _clean_label(atom_name, "atom name"): _normalize_xyz(position, f"position for atom {atom_name}")
            for atom_name, position in self.atom_positions.items()
        }
        object.__setattr__(self, "cell_index", tuple(int(value) for value in self.cell_index))
        object.__setattr__(self, "atom_positions", normalized_positions)
        if "C1" not in normalized_positions or "O4" not in normalized_positions:
            raise ValueError("CelluloseCrystalResidue requires at least C1 and O4 atom positions.")

    def get_atom(self, atom_name: str) -> Tuple[float, float, float]:
        name = _clean_label(atom_name, "atom_name")
        try:
            return self.atom_positions[name]
        except KeyError as exc:
            raise KeyError(f"Atom {name!r} not present in cellulose residue placement.") from exc


@dataclass(frozen=True)
class CelluloseCrystalChain:
    """Ordered cellulose chain reconstructed from the crystal lattice."""

    chain_id: str
    residues: Tuple[CelluloseCrystalResidue, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", _clean_label(self.chain_id, "chain_id"))
        if len(self.chain_id) != 1:
            raise ValueError("CelluloseCrystalChain chain_id must be a single character.")
        if not self.residues:
            raise ValueError("CelluloseCrystalChain requires at least one residue.")


@dataclass(frozen=True)
class PETCrystalResidue:
    """Single PET repeat reconstructed from the WIMZEX01 crystal basis."""

    repeat_index: int
    cell_indices: Tuple[Tuple[int, int, int], ...]
    atom_positions: Dict[str, Tuple[float, float, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.repeat_index < 1:
            raise ValueError("repeat_index must be at least 1.")
        normalized_positions = {
            _clean_label(atom_name, "atom name"): _normalize_xyz(position, f"position for atom {atom_name}")
            for atom_name, position in self.atom_positions.items()
        }
        if not normalized_positions:
            raise ValueError("PETCrystalResidue requires at least one atom position.")
        normalized_cells = []
        for cell_index in self.cell_indices:
            if len(cell_index) != 3:
                raise ValueError("cell_indices entries must contain exactly three integers.")
            normalized_cells.append(tuple(int(value) for value in cell_index))
        object.__setattr__(self, "cell_indices", tuple(normalized_cells))
        object.__setattr__(self, "atom_positions", normalized_positions)

    def get_atom(self, atom_name: str) -> Tuple[float, float, float]:
        name = _clean_label(atom_name, "atom_name")
        try:
            return self.atom_positions[name]
        except KeyError as exc:
            raise KeyError(f"Atom {name!r} not present in PET crystal residue placement.") from exc


@dataclass(frozen=True)
class PETCrystalChain:
    """Ordered PET chain reconstructed from the WIMZEX01 crystal lattice."""

    chain_id: str
    residues: Tuple[PETCrystalResidue, ...]
    bonds: Tuple[Tuple[Tuple[int, str], Tuple[int, str]], ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", _clean_label(self.chain_id, "chain_id"))
        if len(self.chain_id) != 1:
            raise ValueError("PETCrystalChain chain_id must be a single character.")
        if not self.residues:
            raise ValueError("PETCrystalChain requires at least one residue.")


@dataclass(frozen=True)
class CelluloseCrystalSurfaceResidue:
    """Residue reference on a selected cellulose crystal surface."""

    chain_id: str
    residue_number: int
    basis_index: int
    cell_index: Tuple[int, int, int]
    center: Tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", _clean_label(self.chain_id, "chain_id"))
        if len(self.chain_id) != 1:
            raise ValueError("CelluloseCrystalSurfaceResidue chain_id must be a single character.")
        if self.residue_number < 1:
            raise ValueError("residue_number must be at least 1.")
        if len(self.cell_index) != 3:
            raise ValueError("cell_index must contain exactly three integers.")
        object.__setattr__(self, "cell_index", tuple(int(value) for value in self.cell_index))
        object.__setattr__(self, "center", _normalize_xyz(self.center, "center"))


@dataclass(frozen=True)
class CelluloseCrystalSurface:
    """Geometric description of one exposed face of a cellulose crystal."""

    face_name: str
    side: str
    surface_axes: Tuple[str, str]
    normal_axis: str
    normal_vector: Tuple[float, float, float]
    bounding_box_min_angstrom: Tuple[float, float, float]
    bounding_box_max_angstrom: Tuple[float, float, float]
    bounding_box_size_angstrom: Tuple[float, float, float]
    crystal_center: Tuple[float, float, float]
    surface_center: Tuple[float, float, float]
    thickness_angstrom: float
    surface_span_angstrom: Tuple[float, float]
    residue_refs: Tuple[CelluloseCrystalSurfaceResidue, ...]

    def __post_init__(self) -> None:
        face_name = _clean_label(self.face_name, "face_name").lower()
        if len(face_name) != 2 or any(axis not in "abc" for axis in face_name) or len(set(face_name)) != 2:
            raise ValueError("face_name must contain two distinct cellulose axes from 'a', 'b', 'c'.")
        side = _clean_label(self.side, "side").lower()
        if side not in {"min", "max"}:
            raise ValueError("side must be either 'min' or 'max'.")
        normal_axis = _clean_label(self.normal_axis, "normal_axis").lower()
        if normal_axis not in "abc":
            raise ValueError("normal_axis must be one of 'a', 'b', 'c'.")
        if normal_axis in face_name:
            raise ValueError("normal_axis must be orthogonal to face_name.")
        if len(self.surface_axes) != 2 or any(axis not in face_name for axis in self.surface_axes):
            raise ValueError("surface_axes must contain the two axes that define the exposed face.")
        object.__setattr__(self, "face_name", face_name)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "surface_axes", tuple(axis.lower() for axis in self.surface_axes))
        object.__setattr__(self, "normal_axis", normal_axis)
        object.__setattr__(self, "normal_vector", _normalize_xyz(self.normal_vector, "normal_vector"))
        object.__setattr__(
            self,
            "bounding_box_min_angstrom",
            _normalize_xyz(self.bounding_box_min_angstrom, "bounding_box_min_angstrom"),
        )
        object.__setattr__(
            self,
            "bounding_box_max_angstrom",
            _normalize_xyz(self.bounding_box_max_angstrom, "bounding_box_max_angstrom"),
        )
        object.__setattr__(
            self,
            "bounding_box_size_angstrom",
            _normalize_xyz(self.bounding_box_size_angstrom, "bounding_box_size_angstrom"),
        )
        object.__setattr__(self, "crystal_center", _normalize_xyz(self.crystal_center, "crystal_center"))
        object.__setattr__(self, "surface_center", _normalize_xyz(self.surface_center, "surface_center"))
        object.__setattr__(self, "thickness_angstrom", float(self.thickness_angstrom))
        if self.thickness_angstrom <= 0:
            raise ValueError("thickness_angstrom must be positive.")
        if len(self.surface_span_angstrom) != 2:
            raise ValueError("surface_span_angstrom must contain exactly two values.")
        object.__setattr__(
            self,
            "surface_span_angstrom",
            (float(self.surface_span_angstrom[0]), float(self.surface_span_angstrom[1])),
        )
        if not self.residue_refs:
            raise ValueError("CelluloseCrystalSurface requires at least one residue reference.")


@dataclass(frozen=True)
class PolymerChainSpec:
    """Single chain placement within a polymer build specification."""

    chain_id: str
    n_units: int
    start_residue: int = 1
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", _clean_label(self.chain_id, "chain_id"))
        if self.n_units < 1:
            raise ValueError("n_units must be at least 1.")
        if self.start_residue < 1:
            raise ValueError("start_residue must be at least 1.")
        if len(self.translation) != 3:
            raise ValueError("translation must contain exactly three coordinates.")
        coords = tuple(float(coord) for coord in self.translation)
        object.__setattr__(self, "translation", coords)


@dataclass(frozen=True)
class PolymerBuildSpec:
    """Validated polymer build request that can be serialized or executed later."""

    template_name: str
    arrangement: str
    chains: Tuple[PolymerChainSpec, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "template_name", _clean_label(self.template_name, "template_name"))
        arrangement = _clean_label(self.arrangement, "arrangement").lower()
        if arrangement not in {"linear", "bundle", "surface_fragment"}:
            raise ValueError("arrangement must be one of: linear, bundle, surface_fragment.")
        object.__setattr__(self, "arrangement", arrangement)
        if not self.chains:
            raise ValueError("PolymerBuildSpec requires at least one chain.")
        chain_ids = [chain.chain_id for chain in self.chains]
        if len(set(chain_ids)) != len(chain_ids):
            raise ValueError("PolymerBuildSpec chain IDs must be unique.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)

    def write_manifest(self, path: Union[str, Path]) -> str:
        """Write the spec to a JSON manifest file and return its absolute path."""
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return str(out_path)


class PolymerBuilder:
    """Registry and factory for reusable polymer build specifications."""

    def __init__(self, templates: Optional[Iterable[PolymerTemplate]] = None) -> None:
        self._templates: Dict[str, PolymerTemplate] = {}
        for template in templates or ():
            self.register_template(template)

    def register_template(self, template: PolymerTemplate, overwrite: bool = False) -> PolymerTemplate:
        """Register ``template`` and return it."""
        if not isinstance(template, PolymerTemplate):
            raise TypeError("template must be a PolymerTemplate instance.")
        key = template.name.lower()
        if key in self._templates and not overwrite:
            raise ValueError(f"Template {template.name!r} is already registered.")
        self._templates[key] = template
        return template

    def available_templates(self) -> Tuple[str, ...]:
        """Return registered template names in sorted order."""
        return tuple(sorted(template.name for template in self._templates.values()))

    def get_template(self, name: str) -> PolymerTemplate:
        """Return the registered template ``name``."""
        key = _clean_label(name, "template name").lower()
        try:
            return self._templates[key]
        except KeyError as exc:
            available = ", ".join(self.available_templates()) or "none"
            raise KeyError(f"Unknown polymer template {name!r}. Available templates: {available}.") from exc

    def build_linear_spec(
        self,
        template: Union[str, PolymerTemplate],
        n_units: int,
        chain_ids: Optional[Sequence[str]] = None,
        n_chains: int = 1,
        start_residue: int = 1,
        chain_spacing_angstrom: float = 6.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PolymerBuildSpec:
        """Create a linear-chain or parallel-chain polymer specification."""
        if n_units < 1:
            raise ValueError("n_units must be at least 1.")
        if chain_spacing_angstrom <= 0:
            raise ValueError("chain_spacing_angstrom must be positive.")
        template_obj = self._resolve_template(template)
        normalized_chain_ids = _normalize_chain_ids(n_chains, chain_ids=chain_ids)

        chains = []
        for idx, chain_id in enumerate(normalized_chain_ids):
            translation = (0.0, float(idx) * float(chain_spacing_angstrom), 0.0)
            chains.append(
                PolymerChainSpec(
                    chain_id=chain_id,
                    n_units=int(n_units),
                    start_residue=int(start_residue),
                    translation=translation,
                )
            )

        arrangement = "linear" if len(chains) == 1 else "bundle"
        combined_metadata: Dict[str, Any] = {
            "template_description": template_obj.description,
            "linkage_atoms": template_obj.linkage_atoms,
            "bond_length_angstrom": template_obj.bond_length_angstrom,
        }
        if metadata:
            combined_metadata.update(dict(metadata))
        return PolymerBuildSpec(
            template_name=template_obj.name,
            arrangement=arrangement,
            chains=tuple(chains),
            metadata=combined_metadata,
        )

    def build_bundle_spec(
        self,
        template: Union[str, PolymerTemplate],
        n_units: int,
        n_chains: int,
        chain_ids: Optional[Sequence[str]] = None,
        start_residue: int = 1,
        chain_spacing_angstrom: float = 6.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PolymerBuildSpec:
        """Create a multi-chain bundle specification."""
        if n_chains < 2:
            raise ValueError("build_bundle_spec requires at least 2 chains.")
        return self.build_linear_spec(
            template=template,
            n_units=n_units,
            chain_ids=chain_ids,
            n_chains=n_chains,
            start_residue=start_residue,
            chain_spacing_angstrom=chain_spacing_angstrom,
            metadata=metadata,
        )

    def build_surface_fragment_spec(
        self,
        template: Union[str, PolymerTemplate],
        n_units: int,
        n_rows: int,
        n_columns: int,
        chain_ids: Optional[Sequence[str]] = None,
        start_residue: int = 1,
        row_spacing_angstrom: float = 6.0,
        column_spacing_angstrom: float = 6.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PolymerBuildSpec:
        """Create a simple grid-like surface-fragment specification."""
        if n_units < 1:
            raise ValueError("n_units must be at least 1.")
        if n_rows < 1 or n_columns < 1:
            raise ValueError("n_rows and n_columns must be at least 1.")
        if row_spacing_angstrom <= 0 or column_spacing_angstrom <= 0:
            raise ValueError("row_spacing_angstrom and column_spacing_angstrom must be positive.")

        template_obj = self._resolve_template(template)
        n_chains = n_rows * n_columns
        normalized_chain_ids = _normalize_chain_ids(n_chains, chain_ids=chain_ids)

        chains = []
        chain_index = 0
        for row in range(n_rows):
            for column in range(n_columns):
                translation = (
                    float(column) * float(column_spacing_angstrom),
                    float(row) * float(row_spacing_angstrom),
                    0.0,
                )
                chains.append(
                    PolymerChainSpec(
                        chain_id=normalized_chain_ids[chain_index],
                        n_units=int(n_units),
                        start_residue=int(start_residue),
                        translation=translation,
                    )
                )
                chain_index += 1

        combined_metadata: Dict[str, Any] = {
            "template_description": template_obj.description,
            "linkage_atoms": template_obj.linkage_atoms,
            "bond_length_angstrom": template_obj.bond_length_angstrom,
            "n_rows": int(n_rows),
            "n_columns": int(n_columns),
        }
        if metadata:
            combined_metadata.update(dict(metadata))
        return PolymerBuildSpec(
            template_name=template_obj.name,
            arrangement="surface_fragment",
            chains=tuple(chains),
            metadata=combined_metadata,
        )

    def build_glycam_codes(
        self,
        n_units: int,
        *,
        sugar_code: str = "G",
        anomer: str = "B",
        linkage_position: Union[int, str] = 4,
        terminal_code: Union[int, str] = 0,
    ) -> Tuple[str, ...]:
        """Build a linear GLYCAM residue code sequence for a linked oligosaccharide.

        The common beta-1,4 glucose case used for cellulose corresponds to:

        - internal/nonreducing residues: ``4GB``
        - reducing-end residue: ``0GB``
        """
        if n_units < 1:
            raise ValueError("n_units must be at least 1.")
        sugar = _clean_label(sugar_code, "sugar_code")
        if len(sugar) != 1:
            raise ValueError("sugar_code must be a single GLYCAM sugar letter.")
        anomer_code = _clean_label(anomer, "anomer").upper()
        if len(anomer_code) != 1:
            raise ValueError("anomer must be a single GLYCAM anomer code.")
        linkage = _clean_label(str(linkage_position), "linkage_position")
        terminal = _clean_label(str(terminal_code), "terminal_code")

        internal_code = f"{linkage}{sugar.upper()}{anomer_code}"
        terminal_code_value = f"{terminal}{sugar.upper()}{anomer_code}"
        if n_units == 1:
            return (terminal_code_value,)
        return tuple([internal_code] * (n_units - 1) + [terminal_code_value])

    def write_glycam_tleap_bundle_script(
        self,
        spec: PolymerBuildSpec,
        output_prefix: Union[str, Path],
        *,
        residue_codes_by_chain: Optional[Mapping[str, Sequence[str]]] = None,
        sugar_code: str = "G",
        anomer: str = "B",
        linkage_position: Union[int, str] = 4,
        terminal_code: Union[int, str] = 0,
        reducing_end_cap: Optional[str] = None,
        leaprc: str = "leaprc.GLYCAM_06j-1",
        script_path: Optional[Union[str, Path]] = None,
        check: bool = True,
        solvate: bool = False,
        solvent_box: str = "TIP3PBOX",
        solvent_buffer_angstrom: float = 12.0,
        neutralize: bool = False,
    ) -> str:
        """Write a tleap input that builds a GLYCAM polymer bundle from ``spec``."""
        prefix = Path(output_prefix).resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)

        chain_code_map: Dict[str, Tuple[str, ...]] = {}
        for chain in spec.chains:
            if residue_codes_by_chain and chain.chain_id in residue_codes_by_chain:
                codes = tuple(str(code) for code in residue_codes_by_chain[chain.chain_id])
            else:
                codes = self.build_glycam_codes(
                    chain.n_units,
                    sugar_code=sugar_code,
                    anomer=anomer,
                    linkage_position=linkage_position,
                    terminal_code=terminal_code,
                )
            if len(codes) != chain.n_units:
                raise ValueError(
                    f"Chain {chain.chain_id!r} expected {chain.n_units} GLYCAM residue codes, got {len(codes)}."
                )
            chain_code_map[chain.chain_id] = codes

        lines = [f"source {leaprc}"]
        chain_var_names = []
        for index, chain in enumerate(spec.chains, start=1):
            var_name = f"chain_{index:02d}"
            chain_var_names.append(var_name)
            sequence_codes = list(chain_code_map[chain.chain_id])
            if reducing_end_cap is not None:
                sequence_codes = [_clean_label(reducing_end_cap, "reducing_end_cap")] + sequence_codes
            code_string = " ".join(sequence_codes)
            lines.append(f"{var_name} = sequence {{ {code_string} }}")
            if any(abs(coord) > 1e-8 for coord in chain.translation):
                x, y, z = chain.translation
                lines.append(f"translate {var_name} {{ {x:.3f} {y:.3f} {z:.3f} }}")

        if len(chain_var_names) == 1:
            system_var = chain_var_names[0]
        else:
            system_var = "polymer_bundle"
            lines.append(f"{system_var} = combine {{ {' '.join(chain_var_names)} }}")

        if check:
            lines.append(f"check {system_var}")
        if solvate:
            lines.append(f"solvateBox {system_var} {solvent_box} {float(solvent_buffer_angstrom):.3f}")
            if neutralize:
                lines.append(f"addIonsRand {system_var} Na+ 0")
                lines.append(f"addIonsRand {system_var} Cl- 0")

        pdb_path = prefix.with_suffix(".pdb")
        prmtop_path = prefix.with_suffix(".prmtop")
        rst7_path = prefix.with_suffix(".rst7")
        lines.append(f"savepdb {system_var} {pdb_path}")
        lines.append(f"saveamberparm {system_var} {prmtop_path} {rst7_path}")
        lines.append("quit")

        if script_path is None:
            script_path = prefix.with_suffix(".tleap.in")
        out_script = Path(script_path).resolve()
        out_script.parent.mkdir(parents=True, exist_ok=True)
        out_script.write_text("\n".join(lines) + "\n")
        return str(out_script)

    def run_tleap(
        self,
        script_path: Union[str, Path],
        *,
        tleap_executable: str = "tleap",
        cwd: Optional[Union[str, Path]] = None,
    ) -> subprocess.CompletedProcess:
        """Execute ``tleap`` on ``script_path`` and return the completed process."""
        script = Path(script_path).resolve()
        if not script.exists():
            raise FileNotFoundError(f"tleap script not found: {script}")
        run_cwd = str(Path(cwd).resolve()) if cwd is not None else str(script.parent)
        return subprocess.run(
            [tleap_executable, "-f", str(script)],
            cwd=run_cwd,
            capture_output=True,
            text=True,
            check=False,
        )

    def build_glycam_bundle(
        self,
        spec: PolymerBuildSpec,
        output_prefix: Union[str, Path],
        *,
        residue_codes_by_chain: Optional[Mapping[str, Sequence[str]]] = None,
        sugar_code: str = "G",
        anomer: str = "B",
        linkage_position: Union[int, str] = 4,
        terminal_code: Union[int, str] = 0,
        reducing_end_cap: Optional[str] = None,
        leaprc: str = "leaprc.GLYCAM_06j-1",
        check: bool = True,
        solvate: bool = False,
        solvent_box: str = "TIP3PBOX",
        solvent_buffer_angstrom: float = 12.0,
        neutralize: bool = False,
        tleap_executable: str = "tleap",
        cwd: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Write and execute a GLYCAM/tleap build for the polymer ``spec``."""
        prefix = Path(output_prefix).resolve()
        script_path = self.write_glycam_tleap_bundle_script(
            spec,
            output_prefix=prefix,
            residue_codes_by_chain=residue_codes_by_chain,
            sugar_code=sugar_code,
            anomer=anomer,
            linkage_position=linkage_position,
            terminal_code=terminal_code,
            reducing_end_cap=reducing_end_cap,
            leaprc=leaprc,
            check=check,
            solvate=solvate,
            solvent_box=solvent_box,
            solvent_buffer_angstrom=solvent_buffer_angstrom,
            neutralize=neutralize,
        )
        result = self.run_tleap(script_path, tleap_executable=tleap_executable, cwd=cwd or prefix.parent)
        output = {
            "script_path": str(Path(script_path).resolve()),
            "pdb_path": str(prefix.with_suffix(".pdb")),
            "prmtop_path": str(prefix.with_suffix(".prmtop")),
            "rst7_path": str(prefix.with_suffix(".rst7")),
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        return output

    def residue_template_from_pdb(
        self,
        pdb_path: Union[str, Path],
        residue_name: Optional[str] = None,
    ) -> PolymerResidueTemplate:
        """Load the first residue from ``pdb_path`` as a reusable residue template."""
        try:
            from Bio.PDB import PDBParser
        except ImportError as exc:
            raise ImportError(
                "Biopython is required to load residue templates from PDB files."
            ) from exc

        path = Path(pdb_path).resolve()
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(path.stem, str(path))

        target_residue = None
        for model in structure:
            for chain in model:
                for residue in chain:
                    hetflag = residue.id[0]
                    if hetflag == "W":
                        continue
                    target_residue = residue
                    break
                if target_residue is not None:
                    break
            if target_residue is not None:
                break

        if target_residue is None:
            raise ValueError(f"No residue found in template PDB {path}.")

        atoms = []
        serial_to_name = {}
        for atom in target_residue.get_atoms():
            atoms.append(
                PolymerAtomTemplate(
                    name=atom.name.strip(),
                    element=(atom.element or _guess_element(atom.name)).strip().title(),
                    position=tuple(float(coord) for coord in atom.coord),
                )
            )
            serial = getattr(atom, "serial_number", None)
            if serial is not None:
                serial_to_name[int(serial)] = atom.name.strip()

        bonds = []
        if serial_to_name:
            seen_bonds = set()
            for line in path.read_text().splitlines():
                if not line.startswith("CONECT"):
                    continue
                fields = (line[6:11], line[11:16], line[16:21], line[21:26], line[26:31])
                serials = [int(field) for field in fields if field.strip()]
                if len(serials) < 2 or serials[0] not in serial_to_name:
                    continue
                atom_a = serial_to_name[serials[0]]
                for other_serial in serials[1:]:
                    if other_serial not in serial_to_name:
                        continue
                    bond_key = tuple(sorted((atom_a, serial_to_name[other_serial])))
                    if bond_key in seen_bonds:
                        continue
                    seen_bonds.add(bond_key)
                    bonds.append((atom_a, serial_to_name[other_serial]))

        template_residue_name = residue_name or target_residue.resname.strip()
        return PolymerResidueTemplate(
            residue_name=template_residue_name,
            atoms=tuple(atoms),
            bonds=tuple(bonds),
            metadata={"source_pdb": str(path)},
        )

    def _resolve_residue_template(
        self,
        residue_template: Union[PolymerResidueTemplate, str, Path],
        *,
        residue_name: Optional[str] = None,
    ) -> PolymerResidueTemplate:
        if isinstance(residue_template, PolymerResidueTemplate):
            return residue_template
        return self.residue_template_from_pdb(residue_template, residue_name=residue_name)

    def build_atom_records(
        self,
        spec: PolymerBuildSpec,
        residue_template: PolymerResidueTemplate,
        repeat_vector_angstrom: Sequence[float],
        origin_angstrom: Sequence[float] = (0.0, 0.0, 0.0),
        residue_mode: str = "by_unit",
    ) -> Tuple[Dict[str, Any], ...]:
        """Expand ``spec`` into atom records using ``residue_template`` coordinates."""
        repeat_vector = _normalize_xyz(repeat_vector_angstrom, "repeat_vector_angstrom")
        origin = _normalize_xyz(origin_angstrom, "origin_angstrom")
        residue_mode_name = _clean_label(residue_mode, "residue_mode").lower()
        if residue_mode_name not in {"by_unit", "by_chain"}:
            raise ValueError("residue_mode must be either 'by_unit' or 'by_chain'.")
        records = []
        serial = 1

        for chain in spec.chains:
            chain_offset = _add_xyz(origin, chain.translation)
            for unit_index in range(chain.n_units):
                residue_offset = _add_xyz(chain_offset, _scale_xyz(repeat_vector, unit_index))
                if residue_mode_name == "by_chain":
                    residue_number = chain.start_residue
                else:
                    residue_number = chain.start_residue + unit_index
                for atom in residue_template.atoms:
                    position = _add_xyz(atom.position, residue_offset)
                    records.append(
                        {
                            "serial": serial,
                            "atom_name": atom.name,
                            "residue_name": residue_template.residue_name,
                            "chain_id": chain.chain_id,
                            "residue_number": residue_number,
                            "x": position[0],
                            "y": position[1],
                            "z": position[2],
                            "element": atom.element,
                        }
                    )
                    serial += 1

        return tuple(records)

    def write_pdb(
        self,
        spec: PolymerBuildSpec,
        residue_template: PolymerResidueTemplate,
        output_pdb: Union[str, Path],
        repeat_vector_angstrom: Sequence[float],
        origin_angstrom: Sequence[float] = (0.0, 0.0, 0.0),
        remark_lines: Optional[Sequence[str]] = None,
        residue_mode: str = "by_unit",
        include_ter_records: bool = False,
        include_conect_records: bool = False,
        cryst1_record: Optional[str] = None,
    ) -> str:
        """Write a simple PDB from ``spec`` and return its absolute path."""
        out_path = Path(output_pdb).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        conect_pairs = []
        if cryst1_record:
            lines.append(str(cryst1_record).rstrip())
        for remark in remark_lines or ():
            lines.append(f"REMARK {remark}")
        serial = 1
        repeat_vector = _normalize_xyz(repeat_vector_angstrom, "repeat_vector_angstrom")
        origin = _normalize_xyz(origin_angstrom, "origin_angstrom")
        residue_mode_name = _clean_label(residue_mode, "residue_mode").lower()
        if residue_mode_name not in {"by_unit", "by_chain"}:
            raise ValueError("residue_mode must be either 'by_unit' or 'by_chain'.")
        for chain in spec.chains:
            chain_offset = _add_xyz(origin, chain.translation)
            for unit_index in range(chain.n_units):
                residue_offset = _add_xyz(chain_offset, _scale_xyz(repeat_vector, unit_index))
                if residue_mode_name == "by_chain":
                    residue_number = chain.start_residue
                else:
                    residue_number = chain.start_residue + unit_index
                residue_serials = {}
                for atom in residue_template.atoms:
                    position = _add_xyz(atom.position, residue_offset)
                    atom_name = atom.name
                    padded_atom_name = atom_name.rjust(4) if len(atom_name) < 4 else atom_name[:4]
                    element = str(atom.element).strip().upper()[:2].rjust(2)
                    line = (
                        f"HETATM{serial:5d} {padded_atom_name} "
                        f"{residue_template.residue_name[:3]:>3s} {chain.chain_id[:1]:1s}"
                        f"{residue_number:4d}    "
                        f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
                        f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
                    )
                    lines.append(line)
                    residue_serials[atom.name] = serial
                    serial += 1
                if include_conect_records:
                    for atom_a, atom_b in residue_template.bonds:
                        conect_pairs.append((residue_serials[atom_a], residue_serials[atom_b]))
            if include_ter_records:
                lines.append("TER")
        if include_conect_records:
            for atom_a_serial, atom_b_serial in conect_pairs:
                lines.append(f"CONECT{atom_a_serial:5d}{atom_b_serial:5d}")
        lines.append("END")
        out_path.write_text("\n".join(lines) + "\n")
        return str(out_path)

    def build_polyethylene_terephthalate_crystal_spec(
        self,
        n_units: int,
        n_rows: int,
        n_columns: int,
        *,
        chain_ids: Optional[Sequence[str]] = None,
        start_residue: int = 1,
        a_spacing_angstrom: float = 4.56,
        b_spacing_angstrom: float = 5.94,
        alternating_z_offset_angstrom: Optional[float] = None,
        repeat_vector_angstrom: Sequence[float] = (0.0, 0.0, 10.75),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PolymerBuildSpec:
        """Create a simple crystal-like PET bundle with alternating row offsets.

        The returned specification uses a regular in-plane grid of parallel chains.
        Neighboring rows are shifted by half a repeat along the chain axis by
        default, which approximates the staggered packing commonly used for PET
        crystal-like models.
        """
        if n_units < 1:
            raise ValueError("n_units must be at least 1.")
        if n_rows < 1 or n_columns < 1:
            raise ValueError("n_rows and n_columns must be at least 1.")
        if a_spacing_angstrom <= 0 or b_spacing_angstrom <= 0:
            raise ValueError("a_spacing_angstrom and b_spacing_angstrom must be positive.")

        repeat_vector = _normalize_xyz(repeat_vector_angstrom, "repeat_vector_angstrom")
        if alternating_z_offset_angstrom is None:
            alternating_z_offset_angstrom = 0.5 * math.sqrt(sum(component * component for component in repeat_vector))
        if alternating_z_offset_angstrom < 0:
            raise ValueError("alternating_z_offset_angstrom must be non-negative.")

        template_obj = _default_polyethylene_terephthalate_template()
        n_chains = n_rows * n_columns
        normalized_chain_ids = _normalize_single_char_chain_ids(n_chains, chain_ids=chain_ids)

        chains = []
        chain_index = 0
        for row in range(n_rows):
            z_offset = float(alternating_z_offset_angstrom) if row % 2 else 0.0
            for column in range(n_columns):
                translation = (
                    float(column) * float(a_spacing_angstrom),
                    float(row) * float(b_spacing_angstrom),
                    z_offset,
                )
                chains.append(
                    PolymerChainSpec(
                        chain_id=normalized_chain_ids[chain_index],
                        n_units=int(n_units),
                        start_residue=int(start_residue),
                        translation=translation,
                    )
                )
                chain_index += 1

        combined_metadata: Dict[str, Any] = {
            "template_description": template_obj.description,
            "linkage_atoms": template_obj.linkage_atoms,
            "bond_length_angstrom": template_obj.bond_length_angstrom,
            "n_rows": int(n_rows),
            "n_columns": int(n_columns),
            "a_spacing_angstrom": float(a_spacing_angstrom),
            "b_spacing_angstrom": float(b_spacing_angstrom),
            "alternating_z_offset_angstrom": float(alternating_z_offset_angstrom),
            "repeat_vector_angstrom": repeat_vector,
            "crystal_like": True,
            "polymer_name": "polyethylene_terephthalate",
        }
        if metadata:
            combined_metadata.update(dict(metadata))
        return PolymerBuildSpec(
            template_name=template_obj.name,
            arrangement="surface_fragment",
            chains=tuple(chains),
            metadata=combined_metadata,
        )

    def _build_pet_wimzex01_supercell_atoms(
        self,
        n_units: int,
        n_a: int,
        n_b: int,
        *,
        a_angstrom: float,
        b_angstrom: float,
        c_angstrom: float,
        alpha_degrees: float,
        beta_degrees: float,
        gamma_degrees: float,
    ) -> Tuple[
        Tuple[Dict[str, Any], ...],
        Dict[Tuple[int, int, int], Tuple[int, ...]],
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]:
        a_vec, b_vec, c_vec = _triclinic_lattice_vectors_with_c_along_z(
            a_angstrom,
            b_angstrom,
            c_angstrom,
            alpha_degrees,
            beta_degrees,
            gamma_degrees,
        )
        unit_cells_a = int(n_a) + 1
        unit_cells_b = int(n_b) + 1
        atoms = []
        cell_atoms: Dict[Tuple[int, int, int], list[int]] = {}
        atom_index = 0
        for a_index in range(unit_cells_a):
            for b_index in range(unit_cells_b):
                for unit_index in range(int(n_units)):
                    cell_index = (a_index, b_index, unit_index)
                    cell_entries = []
                    for atom_name, element, frac in _PET_WIMZEX01_ATOM_SITES:
                        shifted_frac = (
                            float(frac[0]) + float(a_index),
                            float(frac[1]) + float(b_index),
                            float(frac[2]) + float(unit_index),
                        )
                        atoms.append(
                            {
                                "index": atom_index,
                                "atom_name": atom_name,
                                "element": element,
                                "cell_index": cell_index,
                                "repeat_index": unit_index + 1,
                                "fractional": shifted_frac,
                                "position": _frac_to_cart_with_lattice_vectors(shifted_frac, a_vec, b_vec, c_vec),
                            }
                        )
                        cell_entries.append(atom_index)
                        atom_index += 1
                    cell_atoms[cell_index] = tuple(cell_entries)
        return tuple(atoms), cell_atoms, a_vec, b_vec, c_vec

    def _infer_pet_wimzex01_bonds(
        self,
        atoms: Sequence[Mapping[str, Any]],
        cell_atoms: Mapping[Tuple[int, int, int], Sequence[int]],
        *,
        bond_tolerance_angstrom: float,
    ) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, ...], ...]]:
        adjacency = [set() for _ in atoms]
        edges = []
        for cell_index, atom_indices in cell_atoms.items():
            for offset in _POSITIVE_NEIGHBOR_CELL_OFFSETS:
                neighbor = (
                    int(cell_index[0]) + int(offset[0]),
                    int(cell_index[1]) + int(offset[1]),
                    int(cell_index[2]) + int(offset[2]),
                )
                other_atom_indices = cell_atoms.get(neighbor)
                if other_atom_indices is None:
                    continue

                if offset == (0, 0, 0):
                    pair_source = []
                    for local_index, atom_a_index in enumerate(atom_indices[:-1]):
                        for atom_b_index in atom_indices[local_index + 1 :]:
                            pair_source.append((atom_a_index, atom_b_index))
                else:
                    pair_source = [
                        (atom_a_index, atom_b_index)
                        for atom_a_index in atom_indices
                        for atom_b_index in other_atom_indices
                    ]

                for atom_a_index, atom_b_index in pair_source:
                    atom_a = atoms[atom_a_index]
                    atom_b = atoms[atom_b_index]
                    radius_a = _PET_WIMZEX01_COVALENT_RADII.get(str(atom_a["element"]).upper())
                    radius_b = _PET_WIMZEX01_COVALENT_RADII.get(str(atom_b["element"]).upper())
                    if radius_a is None or radius_b is None:
                        continue
                    cutoff = float(radius_a) + float(radius_b) + float(bond_tolerance_angstrom)
                    if math.dist(atom_a["position"], atom_b["position"]) > cutoff:
                        continue
                    adjacency[atom_a_index].add(atom_b_index)
                    adjacency[atom_b_index].add(atom_a_index)
                    edges.append((atom_a_index, atom_b_index))

        return tuple(edges), tuple(tuple(sorted(neighbors)) for neighbors in adjacency)

    def build_polyethylene_terephthalate_wimzex01_crystal(
        self,
        n_units: int,
        n_a: int,
        n_b: int,
        *,
        chain_ids: Optional[Sequence[str]] = None,
        a_angstrom: float = 4.56,
        b_angstrom: float = 5.94,
        c_angstrom: float = 10.75,
        alpha_degrees: float = 98.5,
        beta_degrees: float = 118.0,
        gamma_degrees: float = 112.0,
        bond_tolerance_angstrom: float = _PET_WIMZEX01_BOND_TOLERANCE_ANGSTROM,
    ) -> Tuple[PETCrystalChain, ...]:
        """Build a finite PET crystal slab from the WIMZEX01 crystallographic basis.

        ``n_a`` and ``n_b`` count complete PET chains in the lateral slab directions.
        The construction pads the replicated unit cells by one in each lateral
        direction, then keeps only connected components that correspond to
        uninterrupted crystallographic chains.
        """
        if n_units < 1:
            raise ValueError("n_units must be at least 1.")
        if n_a < 1 or n_b < 1:
            raise ValueError("n_a and n_b must both be at least 1.")
        if bond_tolerance_angstrom <= 0.0:
            raise ValueError("bond_tolerance_angstrom must be positive.")

        atoms, cell_atoms, a_vec, b_vec, c_vec = self._build_pet_wimzex01_supercell_atoms(
            n_units=n_units,
            n_a=n_a,
            n_b=n_b,
            a_angstrom=a_angstrom,
            b_angstrom=b_angstrom,
            c_angstrom=c_angstrom,
            alpha_degrees=alpha_degrees,
            beta_degrees=beta_degrees,
            gamma_degrees=gamma_degrees,
        )
        edges, adjacency = self._infer_pet_wimzex01_bonds(
            atoms,
            cell_atoms,
            bond_tolerance_angstrom=bond_tolerance_angstrom,
        )

        visited: set[int] = set()
        components = []
        for atom_index in range(len(atoms)):
            if atom_index in visited:
                continue
            stack = [atom_index]
            component = []
            visited.add(atom_index)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
            components.append(tuple(sorted(component)))

        full_chain_atom_count = len(_PET_WIMZEX01_ATOM_SITES) * int(n_units)
        full_components = [component for component in components if len(component) == full_chain_atom_count]
        expected_chains = int(n_a) * int(n_b)
        if len(full_components) != expected_chains:
            raise ValueError(
                "Failed to recover the expected number of complete PET crystal chains "
                f"({len(full_components)} found, expected {expected_chains})."
            )

        full_components.sort(
            key=lambda component: (
                sum(float(atoms[idx]["position"][1]) for idx in component) / len(component),
                sum(float(atoms[idx]["position"][0]) for idx in component) / len(component),
                sum(float(atoms[idx]["position"][2]) for idx in component) / len(component),
            )
        )
        normalized_chain_ids = _normalize_single_char_chain_ids(expected_chains, chain_ids=chain_ids)
        atom_order_index = {name: order for order, name in enumerate(_PET_WIMZEX01_ATOM_NAME_ORDER)}

        chains = []
        for chain_id, component in zip(normalized_chain_ids, full_components):
            component_set = set(component)
            residues = []
            atom_ref_by_index: Dict[int, Tuple[int, str]] = {}
            for repeat_index in range(1, int(n_units) + 1):
                repeat_atoms = [atoms[idx] for idx in component if int(atoms[idx]["repeat_index"]) == repeat_index]
                if len(repeat_atoms) != len(_PET_WIMZEX01_ATOM_SITES):
                    raise ValueError(
                        f"PET crystal chain {chain_id} repeat {repeat_index} is incomplete: "
                        f"{len(repeat_atoms)} atoms found."
                    )
                residue_positions = {}
                residue_cells = sorted({tuple(atom["cell_index"]) for atom in repeat_atoms})
                for atom_name in _PET_WIMZEX01_ATOM_NAME_ORDER:
                    matches = [atom for atom in repeat_atoms if atom["atom_name"] == atom_name]
                    if len(matches) != 1:
                        raise ValueError(
                            f"PET crystal chain {chain_id} repeat {repeat_index} has "
                            f"{len(matches)} occurrences of atom {atom_name!r}."
                        )
                    residue_positions[atom_name] = matches[0]["position"]
                    atom_ref_by_index[int(matches[0]["index"])] = (repeat_index, atom_name)

                residues.append(
                    PETCrystalResidue(
                        repeat_index=repeat_index,
                        cell_indices=tuple(residue_cells),
                        atom_positions=residue_positions,
                        metadata={
                            "source_refcode": "WIMZEX01",
                            "source_doi": "10.1098/rspa.1954.0273",
                        },
                    )
                )

            bond_refs = []
            seen_bond_refs = set()
            for atom_a_index, atom_b_index in edges:
                if atom_a_index not in component_set or atom_b_index not in component_set:
                    continue
                ref_a = atom_ref_by_index[atom_a_index]
                ref_b = atom_ref_by_index[atom_b_index]
                if ref_a == ref_b:
                    continue
                bond_key = tuple(
                    sorted(
                        (ref_a, ref_b),
                        key=lambda ref: (int(ref[0]), atom_order_index.get(ref[1], 999), str(ref[1])),
                    )
                )
                if bond_key in seen_bond_refs:
                    continue
                seen_bond_refs.add(bond_key)
                bond_refs.append(bond_key)

            center = (
                sum(float(atoms[idx]["position"][0]) for idx in component) / len(component),
                sum(float(atoms[idx]["position"][1]) for idx in component) / len(component),
                sum(float(atoms[idx]["position"][2]) for idx in component) / len(component),
            )
            chains.append(
                PETCrystalChain(
                    chain_id=chain_id,
                    residues=tuple(residues),
                    bonds=tuple(
                        sorted(
                            bond_refs,
                            key=lambda edge: (
                                int(edge[0][0]),
                                atom_order_index.get(edge[0][1], 999),
                                int(edge[1][0]),
                                atom_order_index.get(edge[1][1], 999),
                            ),
                        )
                    ),
                    metadata={
                        "source_refcode": "WIMZEX01",
                        "source_doi": "10.1098/rspa.1954.0273",
                        "n_units": int(n_units),
                        "n_a": int(n_a),
                        "n_b": int(n_b),
                        "cell_parameters_angstrom": {
                            "a": float(a_angstrom),
                            "b": float(b_angstrom),
                            "c": float(c_angstrom),
                        },
                        "cell_angles_degrees": {
                            "alpha": float(alpha_degrees),
                            "beta": float(beta_degrees),
                            "gamma": float(gamma_degrees),
                        },
                        "lattice_vectors_angstrom": {
                            "a": a_vec,
                            "b": b_vec,
                            "c": c_vec,
                        },
                        "chain_center_angstrom": center,
                        "cell_footprint_ab": tuple(
                            sorted({(int(atoms[idx]["cell_index"][0]), int(atoms[idx]["cell_index"][1])) for idx in component})
                        ),
                    },
                )
            )

        return tuple(chains)

    def write_polyethylene_terephthalate_wimzex01_pdb(
        self,
        output_pdb: Union[str, Path],
        *,
        n_units: int,
        n_a: int,
        n_b: int,
        chain_ids: Optional[Sequence[str]] = None,
        start_residue: int = 1,
        a_angstrom: float = 4.56,
        b_angstrom: float = 5.94,
        c_angstrom: float = 10.75,
        alpha_degrees: float = 98.5,
        beta_degrees: float = 118.0,
        gamma_degrees: float = 112.0,
        bond_tolerance_angstrom: float = _PET_WIMZEX01_BOND_TOLERANCE_ANGSTROM,
        include_ter_records: bool = True,
        include_conect_records: bool = False,
        origin_angstrom: Sequence[float] = (0.0, 0.0, 0.0),
        remark_lines: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Write a heavy-atom PET slab reconstructed from the WIMZEX01 CIF basis."""
        chains = self.build_polyethylene_terephthalate_wimzex01_crystal(
            n_units=n_units,
            n_a=n_a,
            n_b=n_b,
            chain_ids=chain_ids,
            a_angstrom=a_angstrom,
            b_angstrom=b_angstrom,
            c_angstrom=c_angstrom,
            alpha_degrees=alpha_degrees,
            beta_degrees=beta_degrees,
            gamma_degrees=gamma_degrees,
            bond_tolerance_angstrom=bond_tolerance_angstrom,
        )

        out_path = Path(output_pdb).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        origin = _normalize_xyz(origin_angstrom, "origin_angstrom")
        cryst1_record = _format_pdb_cryst1_line(
            float(a_angstrom) * (int(n_a) + 1),
            float(b_angstrom) * (int(n_b) + 1),
            float(c_angstrom) * int(n_units),
            alpha_degrees,
            beta_degrees,
            gamma_degrees,
            space_group="P -1",
        )
        lines = [cryst1_record]
        default_remarks = (
            "generated with prepare_proteins PET WIMZEX01 crystal builder",
            (
                f"pet_wimzex01_slab n_units={int(n_units)} n_a={int(n_a)} "
                f"n_b={int(n_b)} n_chains={len(chains)}"
            ),
            "crystallographic heavy-atom basis from CSD refcode WIMZEX01",
            "original PET crystal reference: Daubeny and Bunn, Proc. R. Soc. A 1954, doi 10.1098/rspa.1954.0273",
            (
                f"unit_cell_padding_for_complete_chains a_cells={int(n_a) + 1} "
                f"b_cells={int(n_b) + 1}"
            ),
        )
        for remark in tuple(default_remarks) + tuple(remark_lines or ()):
            lines.append(f"REMARK {remark}")

        serial = 1
        atom_serials: Dict[Tuple[str, int, str], int] = {}
        atom_count = 0
        for chain in chains:
            for residue in chain.residues:
                residue_number = int(start_residue) + int(residue.repeat_index) - 1
                for atom_name in _PET_WIMZEX01_ATOM_NAME_ORDER:
                    position = _add_xyz(residue.get_atom(atom_name), origin)
                    lines.append(
                        _format_pdb_atom_line(
                            serial,
                            atom_name,
                            "PET",
                            chain.chain_id,
                            residue_number,
                            position,
                            _guess_element(atom_name),
                        )
                    )
                    atom_serials[(chain.chain_id, residue.repeat_index, atom_name)] = serial
                    serial += 1
                    atom_count += 1
            if include_ter_records:
                lines.append("TER")

        if include_conect_records:
            conect_pairs = []
            seen_pairs = set()
            for chain in chains:
                for bond_a, bond_b in chain.bonds:
                    serial_a = atom_serials[(chain.chain_id, int(bond_a[0]), str(bond_a[1]))]
                    serial_b = atom_serials[(chain.chain_id, int(bond_b[0]), str(bond_b[1]))]
                    pair = (min(serial_a, serial_b), max(serial_a, serial_b))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    conect_pairs.append(pair)
            for serial_a, serial_b in sorted(conect_pairs):
                lines.append(f"CONECT{serial_a:5d}{serial_b:5d}")

        lines.append("END")
        out_path.write_text("\n".join(lines) + "\n")
        return {
            "pdb_path": str(out_path),
            "n_chains": len(chains),
            "chain_lengths": tuple(len(chain.residues) for chain in chains),
            "n_units": int(n_units),
            "n_a": int(n_a),
            "n_b": int(n_b),
            "n_atoms": int(atom_count),
            "unit_cells_a": int(n_a) + 1,
            "unit_cells_b": int(n_b) + 1,
            "a_angstrom": float(a_angstrom),
            "b_angstrom": float(b_angstrom),
            "c_angstrom": float(c_angstrom),
            "alpha_degrees": float(alpha_degrees),
            "beta_degrees": float(beta_degrees),
            "gamma_degrees": float(gamma_degrees),
            "bond_tolerance_angstrom": float(bond_tolerance_angstrom),
            "crystal_model": "WIMZEX01_cif_basis",
            "source_refcode": "WIMZEX01",
            "include_ter_records": bool(include_ter_records),
            "include_conect_records": bool(include_conect_records),
        }

    def build_polyethylene_terephthalate_triclinic_crystal_spec(
        self,
        n_units: int,
        n_a: int,
        n_b: int,
        *,
        chain_ids: Optional[Sequence[str]] = None,
        start_residue: int = 1,
        a_angstrom: float = 4.56,
        b_angstrom: float = 5.94,
        c_angstrom: float = 10.75,
        alpha_degrees: float = 98.5,
        beta_degrees: float = 118.0,
        gamma_degrees: float = 112.0,
        repeat_vector_angstrom: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PolymerBuildSpec:
        """Create a PET crystal slab using the published triclinic unit-cell geometry.

        This builder places one parallel PET chain on each replicated lattice site.
        The resulting slab follows the triclinic cell vectors, analogous to the
        explicit lattice handling used for the cellulose Ibeta builder, but does
        not attempt a full atomic crystallographic basis reconstruction.
        """
        if n_units < 1:
            raise ValueError("n_units must be at least 1.")
        if n_a < 1 or n_b < 1:
            raise ValueError("n_a and n_b must be at least 1.")

        a_vec, b_vec, c_vec = _triclinic_lattice_vectors_with_c_along_z(
            a_angstrom,
            b_angstrom,
            c_angstrom,
            alpha_degrees,
            beta_degrees,
            gamma_degrees,
        )
        repeat_vector = c_vec if repeat_vector_angstrom is None else _normalize_xyz(
            repeat_vector_angstrom,
            "repeat_vector_angstrom",
        )

        template_obj = _default_polyethylene_terephthalate_template()
        n_chains = n_a * n_b
        normalized_chain_ids = _normalize_single_char_chain_ids(n_chains, chain_ids=chain_ids)

        chains = []
        chain_index = 0
        for b_index in range(n_b):
            b_offset = _scale_xyz(b_vec, b_index)
            for a_index in range(n_a):
                translation = _add_xyz(_scale_xyz(a_vec, a_index), b_offset)
                chains.append(
                    PolymerChainSpec(
                        chain_id=normalized_chain_ids[chain_index],
                        n_units=int(n_units),
                        start_residue=int(start_residue),
                        translation=translation,
                    )
                )
                chain_index += 1

        combined_metadata: Dict[str, Any] = {
            "template_description": template_obj.description,
            "linkage_atoms": template_obj.linkage_atoms,
            "bond_length_angstrom": template_obj.bond_length_angstrom,
            "n_a": int(n_a),
            "n_b": int(n_b),
            "a_angstrom": float(a_angstrom),
            "b_angstrom": float(b_angstrom),
            "c_angstrom": float(c_angstrom),
            "alpha_degrees": float(alpha_degrees),
            "beta_degrees": float(beta_degrees),
            "gamma_degrees": float(gamma_degrees),
            "repeat_vector_angstrom": repeat_vector,
            "lattice_vectors_angstrom": {
                "a": a_vec,
                "b": b_vec,
                "c": c_vec,
            },
            "crystal_like": True,
            "polymer_name": "polyethylene_terephthalate",
            "crystal_model": "triclinic_lattice_parameter_approximation",
        }
        if metadata:
            combined_metadata.update(dict(metadata))
        return PolymerBuildSpec(
            template_name=template_obj.name,
            arrangement="surface_fragment",
            chains=tuple(chains),
            metadata=combined_metadata,
        )

    def write_polyethylene_terephthalate_crystal_pdb(
        self,
        output_pdb: Union[str, Path],
        *,
        residue_template: Union[PolymerResidueTemplate, str, Path],
        n_units: int,
        n_rows: int,
        n_columns: int,
        chain_ids: Optional[Sequence[str]] = None,
        start_residue: int = 1,
        repeat_vector_angstrom: Sequence[float] = (0.0, 0.0, 10.75),
        a_spacing_angstrom: float = 4.56,
        b_spacing_angstrom: float = 5.94,
        alternating_z_offset_angstrom: Optional[float] = None,
        residue_mode: str = "by_chain",
        include_ter_records: bool = True,
        include_conect_records: bool = False,
        origin_angstrom: Sequence[float] = (0.0, 0.0, 0.0),
        remark_lines: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Write a crystal-like PET PDB and return a summary dictionary."""
        repeat_vector = _normalize_xyz(repeat_vector_angstrom, "repeat_vector_angstrom")
        template = self._resolve_residue_template(residue_template, residue_name="PET")
        spec = self.build_polyethylene_terephthalate_crystal_spec(
            n_units=n_units,
            n_rows=n_rows,
            n_columns=n_columns,
            chain_ids=chain_ids,
            start_residue=start_residue,
            a_spacing_angstrom=a_spacing_angstrom,
            b_spacing_angstrom=b_spacing_angstrom,
            alternating_z_offset_angstrom=alternating_z_offset_angstrom,
            repeat_vector_angstrom=repeat_vector_angstrom,
        )
        default_remarks = (
            "generated with prepare_proteins PET crystal-like builder",
            (
                f"pet_crystal_like n_units={int(n_units)} n_rows={int(n_rows)} "
                f"n_columns={int(n_columns)} n_chains={len(spec.chains)}"
            ),
            (
                f"repeat_vector_angstrom ({repeat_vector[0]:.3f}, "
                f"{repeat_vector[1]:.3f}, {repeat_vector[2]:.3f})"
            ),
            (
                f"chain_spacing_angstrom a={float(a_spacing_angstrom):.3f} "
                f"b={float(b_spacing_angstrom):.3f}"
            ),
        )
        pdb_path = self.write_pdb(
            spec,
            residue_template=template,
            output_pdb=output_pdb,
            repeat_vector_angstrom=repeat_vector_angstrom,
            origin_angstrom=origin_angstrom,
            remark_lines=tuple(default_remarks) + tuple(remark_lines or ()),
            residue_mode=residue_mode,
            include_ter_records=include_ter_records,
            include_conect_records=include_conect_records,
        )
        return {
            "pdb_path": pdb_path,
            "n_chains": len(spec.chains),
            "chain_lengths": tuple(chain.n_units for chain in spec.chains),
            "n_units": int(n_units),
            "n_rows": int(n_rows),
            "n_columns": int(n_columns),
            "repeat_vector_angstrom": repeat_vector,
            "residue_mode": _clean_label(residue_mode, "residue_mode").lower(),
            "include_ter_records": bool(include_ter_records),
            "include_conect_records": bool(include_conect_records),
        }

    def write_polyethylene_terephthalate_triclinic_crystal_pdb(
        self,
        output_pdb: Union[str, Path],
        *,
        residue_template: Union[PolymerResidueTemplate, str, Path],
        n_units: int,
        n_a: int,
        n_b: int,
        chain_ids: Optional[Sequence[str]] = None,
        start_residue: int = 1,
        a_angstrom: float = 4.56,
        b_angstrom: float = 5.94,
        c_angstrom: float = 10.75,
        alpha_degrees: float = 98.5,
        beta_degrees: float = 118.0,
        gamma_degrees: float = 112.0,
        repeat_vector_angstrom: Optional[Sequence[float]] = None,
        residue_mode: str = "by_chain",
        include_ter_records: bool = True,
        include_conect_records: bool = False,
        origin_angstrom: Sequence[float] = (0.0, 0.0, 0.0),
        remark_lines: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Write a PET slab using the published triclinic unit-cell parameters."""
        a_vec, b_vec, c_vec = _triclinic_lattice_vectors_with_c_along_z(
            a_angstrom,
            b_angstrom,
            c_angstrom,
            alpha_degrees,
            beta_degrees,
            gamma_degrees,
        )
        repeat_vector = c_vec if repeat_vector_angstrom is None else _normalize_xyz(
            repeat_vector_angstrom,
            "repeat_vector_angstrom",
        )
        template = self._resolve_residue_template(residue_template, residue_name="PET")
        spec = self.build_polyethylene_terephthalate_triclinic_crystal_spec(
            n_units=n_units,
            n_a=n_a,
            n_b=n_b,
            chain_ids=chain_ids,
            start_residue=start_residue,
            a_angstrom=a_angstrom,
            b_angstrom=b_angstrom,
            c_angstrom=c_angstrom,
            alpha_degrees=alpha_degrees,
            beta_degrees=beta_degrees,
            gamma_degrees=gamma_degrees,
            repeat_vector_angstrom=repeat_vector,
        )
        cryst1_record = _format_pdb_cryst1_line(
            float(a_angstrom) * int(n_a),
            float(b_angstrom) * int(n_b),
            float(c_angstrom) * int(n_units),
            alpha_degrees,
            beta_degrees,
            gamma_degrees,
        )
        default_remarks = (
            "generated with prepare_proteins PET triclinic crystal builder",
            (
                f"pet_triclinic_slab n_units={int(n_units)} n_a={int(n_a)} "
                f"n_b={int(n_b)} n_chains={len(spec.chains)}"
            ),
            (
                f"cell_parameters_angstrom a={float(a_angstrom):.3f} "
                f"b={float(b_angstrom):.3f} c={float(c_angstrom):.3f}"
            ),
            (
                f"cell_angles_degrees alpha={float(alpha_degrees):.3f} "
                f"beta={float(beta_degrees):.3f} gamma={float(gamma_degrees):.3f}"
            ),
            (
                f"lattice_vector_a_angstrom ({a_vec[0]:.3f}, {a_vec[1]:.3f}, {a_vec[2]:.3f})"
            ),
            (
                f"lattice_vector_b_angstrom ({b_vec[0]:.3f}, {b_vec[1]:.3f}, {b_vec[2]:.3f})"
            ),
            "triclinic slab follows published PET lattice parameters; atomic basis remains template-driven",
        )
        pdb_path = self.write_pdb(
            spec,
            residue_template=template,
            output_pdb=output_pdb,
            repeat_vector_angstrom=repeat_vector,
            origin_angstrom=origin_angstrom,
            remark_lines=tuple(default_remarks) + tuple(remark_lines or ()),
            residue_mode=residue_mode,
            include_ter_records=include_ter_records,
            include_conect_records=include_conect_records,
            cryst1_record=cryst1_record,
        )
        return {
            "pdb_path": pdb_path,
            "n_chains": len(spec.chains),
            "chain_lengths": tuple(chain.n_units for chain in spec.chains),
            "n_units": int(n_units),
            "n_a": int(n_a),
            "n_b": int(n_b),
            "repeat_vector_angstrom": repeat_vector,
            "a_angstrom": float(a_angstrom),
            "b_angstrom": float(b_angstrom),
            "c_angstrom": float(c_angstrom),
            "alpha_degrees": float(alpha_degrees),
            "beta_degrees": float(beta_degrees),
            "gamma_degrees": float(gamma_degrees),
            "lattice_vectors_angstrom": {
                "a": a_vec,
                "b": b_vec,
                "c": c_vec,
            },
            "residue_mode": _clean_label(residue_mode, "residue_mode").lower(),
            "include_ter_records": bool(include_ter_records),
            "include_conect_records": bool(include_conect_records),
        }

    def build_cellulose_ibeta_crystal(
        self,
        n_a: int,
        n_b: int,
        n_cellobiose: int,
        *,
        chain_ids: Optional[Sequence[str]] = None,
        linkage_cutoff_angstrom: float = 1.8,
    ) -> Tuple[CelluloseCrystalChain, ...]:
        """Build an ordered cellulose Ibeta crystallite from the crystallographic basis.

        The construction follows the Ibeta crystal basis used by cellulose-builder,
        but reconstructs chains directly in Python by tracing O4-to-C1 linkages
        inside the finite lattice.
        """
        if n_a < 2 or n_b < 2:
            raise ValueError("n_a and n_b must both be at least 2 for a finite Ibeta crystallite.")
        if n_cellobiose < 1:
            raise ValueError("n_cellobiose must be at least 1.")
        if linkage_cutoff_angstrom <= 0:
            raise ValueError("linkage_cutoff_angstrom must be positive.")

        basis_residues = _cellulose_ibeta_fractional_residues()
        layer_residues: list[CelluloseCrystalResidue] = []
        for j in range(1, n_b + 1):
            for i in range(1, n_a + 1):
                keep = {1, 2, 3, 4}
                if i == 1 and j != n_b:
                    keep.discard(4)
                if j == 1 and i != 1 and i != n_a:
                    keep.discard(4)
                if i == n_a and j != 1:
                    keep.discard(2)
                if j == n_b and i != 1 and i != n_a:
                    keep.discard(2)
                if i == 1 and j == n_b:
                    keep.discard(2)
                    keep.discard(4)
                if j == 1 and i == n_a:
                    keep.discard(2)
                    keep.discard(4)

                for basis_index in sorted(keep):
                    atom_positions: Dict[str, Tuple[float, float, float]] = {}
                    for atom_name, frac in zip(_CELLULOSE_IBETA_ATOM_NAMES, basis_residues[basis_index - 1]):
                        renamed_atom_name = _CELLULOSE_IBETA_GLYCAM_ATOM_RENAMES.get(atom_name, atom_name)
                        shifted_frac = (frac[0] + (i - 1), frac[1] + (j - 1), frac[2])
                        atom_positions[renamed_atom_name] = _cellulose_ibeta_frac_to_cart(shifted_frac)
                    layer_residues.append(
                        CelluloseCrystalResidue(
                            basis_index=basis_index,
                            cell_index=(i, j, 1),
                            atom_positions=atom_positions,
                            metadata={"allomorph": "I_beta", "layer_index": 1},
                        )
                    )

        lattice_residues: list[CelluloseCrystalResidue] = []
        for k in range(1, n_cellobiose + 1):
            shift_z = float(k - 1) * _CELLULOSE_IBETA_C
            for residue in layer_residues:
                atom_positions = {
                    atom_name: (coords[0], coords[1], coords[2] + shift_z)
                    for atom_name, coords in residue.atom_positions.items()
                }
                lattice_residues.append(
                    CelluloseCrystalResidue(
                        basis_index=residue.basis_index,
                        cell_index=(residue.cell_index[0], residue.cell_index[1], k),
                        atom_positions=atom_positions,
                        metadata={"allomorph": "I_beta", "cellobiose_repeat": k},
                    )
                )

        outgoing: Dict[int, int] = {}
        incoming: Dict[int, int] = {}
        for index, residue in enumerate(lattice_residues):
            o4_position = residue.get_atom("O4")
            hits = []
            for other_index, other in enumerate(lattice_residues):
                if index == other_index:
                    continue
                distance = math.dist(o4_position, other.get_atom("C1"))
                if distance <= linkage_cutoff_angstrom:
                    hits.append((distance, other_index))
            hits.sort(key=lambda item: item[0])
            if not hits:
                continue
            if len(hits) > 1:
                raise ValueError(
                    f"Ambiguous O4->C1 linkage assignment for lattice residue {index}: {hits[:3]}."
                )
            _, other_index = hits[0]
            outgoing[index] = other_index
            if other_index in incoming:
                raise ValueError(f"Lattice residue {other_index} has multiple incoming glycosidic linkages.")
            incoming[other_index] = index

        starters = [index for index in range(len(lattice_residues)) if index not in incoming]
        if not starters:
            raise ValueError("Failed to identify any reducing-end starts in the Ibeta crystallite.")

        starter_order = sorted(
            starters,
            key=lambda index: (
                lattice_residues[index].get_atom("C1")[1],
                lattice_residues[index].get_atom("C1")[0],
                lattice_residues[index].get_atom("C1")[2],
                lattice_residues[index].basis_index,
            ),
        )
        normalized_chain_ids = _normalize_single_char_chain_ids(len(starter_order), chain_ids=chain_ids)

        visited: set[int] = set()
        chains: list[CelluloseCrystalChain] = []
        for chain_id, start_index in zip(normalized_chain_ids, starter_order):
            chain_residues = []
            current = start_index
            local_seen: set[int] = set()
            while True:
                if current in local_seen:
                    raise ValueError(f"Cycle detected while reconstructing cellulose chain {chain_id}.")
                if current in visited:
                    raise ValueError(f"Lattice residue {current} was assigned to more than one cellulose chain.")
                local_seen.add(current)
                visited.add(current)
                chain_residues.append(lattice_residues[current])
                if current not in outgoing:
                    break
                current = outgoing[current]

            chains.append(
                CelluloseCrystalChain(
                    chain_id=chain_id,
                    residues=tuple(chain_residues),
                    metadata={
                        "allomorph": "I_beta",
                        "n_a": int(n_a),
                        "n_b": int(n_b),
                        "n_cellobiose": int(n_cellobiose),
                    },
                )
            )

        if len(visited) != len(lattice_residues):
            missing = sorted(set(range(len(lattice_residues))) - visited)
            raise ValueError(f"Failed to assign all lattice residues to crystal chains: {missing}.")

        return tuple(chains)

    def describe_cellulose_ibeta_surface(
        self,
        chains: Sequence[CelluloseCrystalChain],
        *,
        exposed_face: str = "bc",
        side: str = "max",
    ) -> CelluloseCrystalSurface:
        """Describe one exposed face of a finite cellulose Ibeta crystallite.

        Parameters
        ----------
        chains
            Ordered crystal chains returned by ``build_cellulose_ibeta_crystal``.
        exposed_face
            Two-axis face label such as ``"bc"`` or ``"ac"``. The omitted axis
            defines the surface normal.
        side
            Which side of the crystallite to select along the surface normal:
            ``"min"`` or ``"max"``.
        """
        if not chains:
            raise ValueError("chains must contain at least one cellulose crystal chain.")

        face_axes = tuple(axis for axis in "abc" if axis in _clean_label(exposed_face, "exposed_face").lower())
        if len(face_axes) != 2:
            raise ValueError("exposed_face must contain exactly two distinct axes from 'a', 'b', 'c'.")
        side_name = _clean_label(side, "side").lower()
        if side_name not in {"min", "max"}:
            raise ValueError("side must be either 'min' or 'max'.")

        axis_to_index = {"a": 0, "b": 1, "c": 2}
        normal_axis = next(axis for axis in "abc" if axis not in face_axes)
        normal_index = axis_to_index[normal_axis]
        surface_axis_indices = tuple(axis_to_index[axis] for axis in face_axes)
        sign = 1.0 if side_name == "max" else -1.0
        normal_vector = [0.0, 0.0, 0.0]
        normal_vector[normal_index] = sign

        all_positions = []
        n_a = chains[0].metadata.get("n_a")
        n_b = chains[0].metadata.get("n_b")
        n_cellobiose = chains[0].metadata.get("n_cellobiose")
        extreme_lookup = {"a": n_a, "b": n_b, "c": n_cellobiose}
        if any(value is None for value in extreme_lookup.values()):
            raise ValueError("Cellulose crystal chain metadata must include n_a, n_b, and n_cellobiose.")
        target_index = 1 if side_name == "min" else int(extreme_lookup[normal_axis])

        residue_refs = []
        for chain in chains:
            for residue_number, residue in enumerate(chain.residues, start=1):
                for position in residue.atom_positions.values():
                    all_positions.append(position)
                if residue.cell_index[normal_index] != target_index:
                    continue
                atom_positions = tuple(residue.atom_positions.values())
                center = tuple(
                    sum(position[axis] for position in atom_positions) / len(atom_positions)
                    for axis in range(3)
                )
                residue_refs.append(
                    CelluloseCrystalSurfaceResidue(
                        chain_id=chain.chain_id,
                        residue_number=residue_number,
                        basis_index=residue.basis_index,
                        cell_index=residue.cell_index,
                        center=center,
                    )
                )

        coords = list(zip(*all_positions))
        min_xyz = tuple(min(values) for values in coords)
        max_xyz = tuple(max(values) for values in coords)
        size_xyz = tuple(maximum - minimum for minimum, maximum in zip(min_xyz, max_xyz))
        crystal_center = tuple((minimum + maximum) * 0.5 for minimum, maximum in zip(min_xyz, max_xyz))
        surface_center = tuple(
            sum(ref.center[axis] for ref in residue_refs) / len(residue_refs)
            for axis in range(3)
        )
        surface_span = tuple(size_xyz[index] for index in surface_axis_indices)
        thickness = size_xyz[normal_index]

        return CelluloseCrystalSurface(
            face_name="".join(face_axes),
            side=side_name,
            surface_axes=face_axes,
            normal_axis=normal_axis,
            normal_vector=tuple(normal_vector),
            bounding_box_min_angstrom=min_xyz,
            bounding_box_max_angstrom=max_xyz,
            bounding_box_size_angstrom=size_xyz,
            crystal_center=crystal_center,
            surface_center=surface_center,
            thickness_angstrom=thickness,
            surface_span_angstrom=surface_span,
            residue_refs=tuple(residue_refs),
        )

    def select_cellulose_ibeta_surface_patch(
        self,
        surface: CelluloseCrystalSurface,
        *,
        max_residues: int = 12,
    ) -> Tuple[CelluloseCrystalSurfaceResidue, ...]:
        """Return the most central residues on a selected cellulose surface."""
        if max_residues < 1:
            raise ValueError("max_residues must be at least 1.")
        if max_residues >= len(surface.residue_refs):
            return surface.residue_refs

        axis_to_index = {"a": 0, "b": 1, "c": 2}
        center = surface.surface_center
        surface_axes = tuple(axis_to_index[axis] for axis in surface.surface_axes)

        def in_plane_distance_squared(ref: CelluloseCrystalSurfaceResidue) -> float:
            return sum((ref.center[index] - center[index]) ** 2 for index in surface_axes)

        ordered = sorted(
            surface.residue_refs,
            key=lambda ref: (
                in_plane_distance_squared(ref),
                ref.cell_index[surface_axes[0]],
                ref.cell_index[surface_axes[1]],
                ref.chain_id,
                ref.residue_number,
            ),
        )
        return tuple(ordered[:max_residues])

    def write_cellulose_ibeta_glycam_pdb(
        self,
        output_pdb: Union[str, Path],
        *,
        n_a: int,
        n_b: int,
        n_cellobiose: int,
        chain_ids: Optional[Sequence[str]] = None,
        template_pdb: Optional[Union[str, Path]] = None,
        linkage_cutoff_angstrom: float = 1.8,
        tleap_executable: str = "tleap",
        remark_lines: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Write a GLYCAM-compatible cellulose Ibeta crystal PDB.

        The crystal backbone coordinates come from the Ibeta crystallographic
        basis. Full GLYCAM atom coordinates are reconstructed by fitting GLYCAM
        residue templates onto each crystal residue, and by projecting the ROH
        reducing-end cap from the fitted first glucose in every chain.
        """
        chains = self.build_cellulose_ibeta_crystal(
            n_a=n_a,
            n_b=n_b,
            n_cellobiose=n_cellobiose,
            chain_ids=chain_ids,
            linkage_cutoff_angstrom=linkage_cutoff_angstrom,
        )

        generated_template = False
        if template_pdb is None:
            with tempfile.TemporaryDirectory(prefix="prepare_proteins_cellulose_template_") as tmpdir:
                template_prefix = Path(tmpdir) / "cellulose_ibeta_template"
                template_result = self.build_glycam_bundle(
                    self.build_linear_spec(_default_cellulose_beta14_template(), n_units=2),
                    template_prefix,
                    sugar_code="G",
                    anomer="B",
                    linkage_position=4,
                    reducing_end_cap="ROH",
                    check=False,
                    tleap_executable=tleap_executable,
                )
                if template_result["returncode"] != 0:
                    raise RuntimeError(
                        "Failed to build temporary GLYCAM cellulose template with tleap:\n"
                        f"{template_result['stdout']}\n{template_result['stderr']}"
                    )
                template_residues = _load_pdb_residues(template_result["pdb_path"])
                generated_template = True
                pdb_summary = self._write_cellulose_ibeta_glycam_pdb_from_templates(
                    output_pdb=output_pdb,
                    chains=chains,
                    template_residues=template_residues,
                    n_a=n_a,
                    n_b=n_b,
                    n_cellobiose=n_cellobiose,
                    remark_lines=remark_lines,
                )
        else:
            template_residues = _load_pdb_residues(template_pdb)
            pdb_summary = self._write_cellulose_ibeta_glycam_pdb_from_templates(
                output_pdb=output_pdb,
                chains=chains,
                template_residues=template_residues,
                n_a=n_a,
                n_b=n_b,
                n_cellobiose=n_cellobiose,
                remark_lines=remark_lines,
            )

        pdb_summary["generated_template"] = generated_template
        pdb_summary["template_pdb"] = None if template_pdb is None else str(Path(template_pdb).resolve())
        return pdb_summary

    def _write_cellulose_ibeta_glycam_pdb_from_templates(
        self,
        *,
        output_pdb: Union[str, Path],
        chains: Sequence[CelluloseCrystalChain],
        template_residues: Sequence[Mapping[str, Any]],
        n_a: int,
        n_b: int,
        n_cellobiose: int,
        remark_lines: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        template_roh = next((residue for residue in template_residues if residue["residue_name"] == "ROH"), None)
        template_4gb = next((residue for residue in template_residues if residue["residue_name"] == "4GB"), None)
        template_0gb = next((residue for residue in template_residues if residue["residue_name"] == "0GB"), None)
        if template_roh is None or template_4gb is None or template_0gb is None:
            raise ValueError("Template PDB must contain ROH, 4GB, and 0GB residues.")

        out_path = Path(output_pdb).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        default_remarks = [
            "generated with prepare_proteins cellulose Ibeta crystal builder",
            "cellulose Ibeta basis from Gomes and Skaf, J Comput Chem 2012, 33, 1338-1346",
            (
                f"lattice_size n_a={int(n_a)} n_b={int(n_b)} "
                f"n_cellobiose={int(n_cellobiose)} n_chains={len(chains)}"
            ),
            f"basis_vector_1 ({_CELLULOSE_IBETA_A:.3f}, 0.000, 0.000)",
            (
                "basis_vector_2 "
                f"({_CELLULOSE_IBETA_B * math.cos(math.radians(_CELLULOSE_IBETA_GAMMA_DEGREES)):.3f}, "
                f"{_CELLULOSE_IBETA_B * math.sin(math.radians(_CELLULOSE_IBETA_GAMMA_DEGREES)):.3f}, 0.000)"
            ),
            f"basis_vector_3 (0.000, 0.000, {_CELLULOSE_IBETA_C:.3f})",
        ]
        for remark in tuple(default_remarks) + tuple(remark_lines or ()):
            lines.append(f"REMARK {remark}")

        serial = 1
        chain_lengths = []
        for chain in chains:
            chain_lengths.append(len(chain.residues))
            residue_number = 1

            first_residue = chain.residues[0]
            reducing_common_atoms = [
                atom_name
                for atom_name in ("C1", "O5", "C2", "C5", "H1")
                if atom_name in template_4gb["atoms"] and atom_name in first_residue.atom_positions
            ]
            if len(reducing_common_atoms) < 3:
                raise ValueError("Insufficient common atoms to position the reducing-end 4GB template.")
            reducing_rotation, reducing_translation = _kabsch_transform(
                [template_4gb["atoms"][atom_name]["position"] for atom_name in reducing_common_atoms],
                [first_residue.atom_positions[atom_name] for atom_name in reducing_common_atoms],
            )
            for atom_name, atom_data in template_roh["atoms"].items():
                position = _apply_transform(atom_data["position"], reducing_rotation, reducing_translation)
                lines.append(
                    _format_pdb_atom_line(
                        serial=serial,
                        atom_name=atom_name,
                        residue_name="ROH",
                        chain_id=chain.chain_id,
                        residue_number=residue_number,
                        position=position,
                        element=atom_data["element"],
                    )
                )
                serial += 1
            residue_number += 1

            for residue_index, target_residue in enumerate(chain.residues):
                is_terminal = residue_index == len(chain.residues) - 1
                template_residue = template_0gb if is_terminal else template_4gb
                common_atoms = [atom_name for atom_name in template_residue["atoms"] if atom_name in target_residue.atom_positions]
                if len(common_atoms) < 3:
                    raise ValueError("Insufficient common atoms to map GLYCAM residue coordinates onto the crystal.")
                if is_terminal:
                    terminal_fit_atoms = [
                        atom_name
                        for atom_name in ("O4", "C4", "C3", "C5")
                        if atom_name in template_residue["atoms"] and atom_name in target_residue.atom_positions
                    ]
                    if len(terminal_fit_atoms) < 3:
                        terminal_fit_atoms = common_atoms
                    rotation, translation = _kabsch_transform(
                        [template_residue["atoms"][atom_name]["position"] for atom_name in terminal_fit_atoms],
                        [target_residue.atom_positions[atom_name] for atom_name in terminal_fit_atoms],
                    )
                for atom_name, atom_data in template_residue["atoms"].items():
                    if atom_name in target_residue.atom_positions:
                        position = target_residue.atom_positions[atom_name]
                    else:
                        if not is_terminal:
                            raise ValueError(
                                f"Internal cellulose residue is missing atom {atom_name!r} required by GLYCAM."
                            )
                        position = _apply_transform(atom_data["position"], rotation, translation)
                    lines.append(
                        _format_pdb_atom_line(
                            serial=serial,
                            atom_name=atom_name,
                            residue_name=template_residue["residue_name"],
                            chain_id=chain.chain_id,
                            residue_number=residue_number,
                            position=position,
                            element=atom_data["element"],
                        )
                    )
                    serial += 1
                residue_number += 1

            lines.append("TER")

        lines.append("END")
        out_path.write_text("\n".join(lines) + "\n")
        return {
            "pdb_path": str(out_path),
            "n_chains": len(chains),
            "chain_lengths": tuple(chain_lengths),
            "n_a": int(n_a),
            "n_b": int(n_b),
            "n_cellobiose": int(n_cellobiose),
        }

    def _resolve_template(self, template: Union[str, PolymerTemplate]) -> PolymerTemplate:
        if isinstance(template, PolymerTemplate):
            return template
        return self.get_template(template)
