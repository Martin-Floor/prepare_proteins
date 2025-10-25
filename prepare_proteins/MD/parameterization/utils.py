"""Shared helpers for parameterization backends."""

from __future__ import annotations

from typing import List, Tuple

_OPENMM_UTILS_IMPORT_ERROR = None
try:  # pragma: no cover - optional dependency
    from openmm import Vec3, unit as u  # type: ignore
    from openmm.app import PDBFile, Topology  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    Vec3 = None  # type: ignore[assignment]
    u = None  # type: ignore[assignment]
    PDBFile = None  # type: ignore[assignment]
    Topology = None  # type: ignore[assignment]
    _OPENMM_UTILS_IMPORT_ERROR = exc
    _OPENMM_UTILS_AVAILABLE = False
else:
    _OPENMM_UTILS_AVAILABLE = True


def _ensure_openmm_utils(feature: str = "OpenMM parameterization utilities") -> None:
    if not _OPENMM_UTILS_AVAILABLE:
        raise ImportError(
            f"OpenMM is required to use {feature}. "
            "Install the 'openmm' package (e.g. `pip install openmm`)."
        ) from _OPENMM_UTILS_IMPORT_ERROR


DEFAULT_PARAMETERIZATION_SKIP_RESIDUES = frozenset({
    "HOH",
    "H2O",
    "WAT",
    "TIP",
    "TIP3",
    "TIP4",
    "TIP5",
    "SPC",
    "SPCE",
    "SOL",
    "NA",
    "CL",
    "K",
    "CA",
    "MG",
    "ZN",
    "CU",
    "MN",
    "FE",
})


def extract_residue_subsystem(modeller, residue) -> Tuple[Topology, List]:
    """Return an OpenMM topology and positions for ``residue`` extracted from ``modeller``."""
    _ensure_openmm_utils("extract_residue_subsystem")
    top = Topology()
    chain = top.addChain(residue.chain.id)
    new_residue = top.addResidue(residue.name, chain, residue.id)

    atom_map = {}
    positions: List[Vec3] = []
    modeller_positions = modeller.getPositions()
    for atom in modeller.topology.atoms():
        if atom.residue == residue:
            new_atom = top.addAtom(atom.name, atom.element, new_residue)
            atom_map[atom.index] = new_atom
            pos = modeller_positions[atom.index]
            if hasattr(pos, "value_in_unit"):
                xyz = pos.value_in_unit(u.nanometer)
                pos = Vec3(xyz[0], xyz[1], xyz[2])
            positions.append(pos)

    if not positions:
        raise ValueError(f"Residue '{residue.name}' has no atoms to extract.")

    for atom1, atom2 in modeller.topology.bonds():
        if atom1.residue == residue and atom2.residue == residue:
            top.addBond(atom_map[atom1.index], atom_map[atom2.index])

    position_quantity = u.Quantity(positions, u.nanometer)
    return top, position_quantity


def write_residue_pdb(modeller, residue, file_path: str) -> Tuple[Topology, List]:
    """Write a residue-only PDB using the modeller coordinates and return topology/positions."""
    _ensure_openmm_utils("write_residue_pdb")
    topology, positions = extract_residue_subsystem(modeller, residue)
    with open(file_path, "w") as handle:
        PDBFile.writeFile(topology, positions, handle)
    return topology, positions
