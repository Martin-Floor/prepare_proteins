from __future__ import annotations

import os
import re
import copy
import glob
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_1to3

_MLCG_IMPORT_ERROR = None
try:  # pragma: no cover - optional dependency
    import mlcg  # type: ignore
    import torch  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    mlcg = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    MLCG_AVAILABLE = False
    _MLCG_IMPORT_ERROR = exc
else:  # pragma: no cover - heavy optional dependency
    MLCG_AVAILABLE = True

KBOLTZMANN = 0.0019872041

_STANDARD_RESNAMES = {v.upper() for v in protein_letters_1to3.values()}
_RESNAME_MAP = {
    "HIE": "HIS",
    "HID": "HIS",
    "HIP": "HIS",
    "ASH": "ASP",
    "GLH": "GLU",
    "CYX": "CYS",
    "MSE": "MET",
}
_ALL_RESIDUES = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]

PAPER_PT_SETTINGS = {
    "2RVD": (200.0, 393.4, 4),
    "2JOF": (200.0, 393.4, 6),
    "1FME": (200.0, 393.4, 6),
    "1YRF": (200.0, 406.8, 8),
    "1ENH": (200.0, 393.5, 11),
    "2A3D": (200.0, 393.5, 11),
    "1RIS": (200.0, 393.5, 11),
    "2NUZ": (200.0, 393.5, 11),
    "2CI2": (200.0, 393.5, 11),
}
DEFAULT_PT_SETTING = (200.0, 393.5, 11)

PAPER_MODEL_ZENODO_RECORD = "15465782"
PAPER_MODEL_ARCHIVE_URL = (
    "https://zenodo.org/api/records/15465782/files/"
    "simulating_a_trained_cg_model.zip/content"
)
PAPER_MODEL_FILENAME = "model_and_prior.pt"


def _ensure_mlcg(feature: str = "MLCG functionality") -> None:
    if not MLCG_AVAILABLE:
        raise ImportError(
            f"MLCG is required to use {feature}. "
            "Install the 'mlcg' package (and its dependencies) to enable this functionality."
    ) from _MLCG_IMPORT_ERROR


def _import_opeps_map():
    try:
        from mlcg.cg import OPEPS_MAP  # type: ignore
    except Exception:
        from mlcg.cg._mappings import OPEPS_MAP  # type: ignore
    return OPEPS_MAP


def _select_atom(residue, atom_name: str):
    best = None
    best_key = (-1, -1.0)
    for atom in residue.get_atoms():
        if atom.name.strip().upper() != atom_name:
            continue
        element = (atom.element or "").strip().upper()
        if element == "H":
            continue
        if not element and atom.name.strip().upper().startswith("H"):
            continue
        altloc = atom.get_altloc()
        occupancy = atom.get_occupancy() or 0.0
        priority = 2 if altloc in ("", " ") else 1 if altloc == "A" else 0
        key = (priority, occupancy)
        if key > best_key:
            best = atom
            best_key = key
    return best


def _guess_pdb_id(model_name: str, pdb_path: Optional[str] = None) -> Optional[str]:
    candidates = []
    if model_name:
        candidates.append(model_name)
    if pdb_path:
        candidates.append(os.path.basename(pdb_path))
    pattern = re.compile(r"\\b[0-9][A-Za-z0-9]{3}\\b")
    for text in candidates:
        match = pattern.search(text)
        if match:
            return match.group(0).upper()
    if pdb_path:
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_path)
            header = getattr(structure, "header", {}) or {}
            pdb_id = header.get("idcode")
            if pdb_id:
                return str(pdb_id).upper()
        except Exception:
            return None
    return None


def _geometric_temperatures(
    tmin: float,
    tmax: float,
    n_replicas: int,
    target: float = 300.0,
) -> List[float]:
    if n_replicas <= 1:
        return [target]
    ratio = (tmax / tmin) ** (1.0 / (n_replicas - 1))
    temps = [tmin * (ratio**i) for i in range(n_replicas)]
    closest = min(range(len(temps)), key=lambda i: abs(temps[i] - target))
    temps[closest] = target
    return sorted(temps)


def pt_betas_from_temperature_range(
    tmin: float,
    tmax: float,
    n_replicas: int,
    target: float = 300.0,
) -> List[float]:
    """Return beta ladder for PT using a geometric temperature progression."""
    temps = _geometric_temperatures(tmin, tmax, n_replicas, target=target)
    return [1.0 / (KBOLTZMANN * t) for t in temps]


def paper_pt_betas(model_name: str, pdb_path: Optional[str] = None) -> List[float]:
    pdb_id = _guess_pdb_id(model_name, pdb_path)
    tmin, tmax, n_replicas = PAPER_PT_SETTINGS.get(pdb_id, DEFAULT_PT_SETTING)
    return pt_betas_from_temperature_range(tmin, tmax, n_replicas, target=300.0)


DEFAULT_MODEL_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "prepare_proteins", "mlcg"
)


def _download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_mlcg_", dir=os.path.dirname(dest_path))
    os.close(fd)
    try:
        with urllib.request.urlopen(url) as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _download_and_extract_zip(
    url: str,
    dest_path: str,
    archive_member: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_mlcg_zip_", dir=os.path.dirname(dest_path))
    os.close(fd)
    try:
        with urllib.request.urlopen(url) as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        member_name = archive_member or os.path.basename(dest_path)
        with zipfile.ZipFile(tmp_path) as zf:
            names = zf.namelist()
            member = member_name if member_name in names else None
            if member is None:
                matches = [n for n in names if os.path.basename(n) == member_name]
                if len(matches) == 1:
                    member = matches[0]
                elif len(matches) > 1:
                    raise ValueError(
                        f"Multiple matches for {member_name} inside archive: {matches}"
                    )
            if member is None:
                raise FileNotFoundError(
                    f"Archive does not contain {member_name}."
                )
            with zf.open(member) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def resolve_pretrained_model_path(
    model_file: str,
    model_url: Optional[str] = None,
    cache_dir: Optional[str] = None,
    archive_member: Optional[str] = None,
) -> str:
    if not model_file:
        raise ValueError("model_file must be provided to locate a pretrained MLCG model.")
    if os.path.isabs(model_file) and os.path.exists(model_file):
        return model_file
    cache_root = os.path.expanduser(cache_dir or DEFAULT_MODEL_CACHE_DIR)
    cached_path = os.path.join(cache_root, os.path.basename(model_file))
    if model_url is None and model_file in (PAPER_MODEL_FILENAME, "model_with_prior.pt"):
        model_url = PAPER_MODEL_ARCHIVE_URL
        archive_member = PAPER_MODEL_FILENAME
    if model_url:
        if os.path.exists(cached_path):
            return cached_path
        if archive_member is None and model_url == PAPER_MODEL_ARCHIVE_URL:
            archive_member = PAPER_MODEL_FILENAME
        if ".zip" in model_url.lower():
            _download_and_extract_zip(model_url, cached_path, archive_member=archive_member)
        else:
            _download_file(model_url, cached_path)
        return cached_path
    if os.path.exists(model_file):
        return os.path.abspath(model_file)
    if os.path.exists(cached_path):
        return cached_path
    raise FileNotFoundError(
        "MLCG model file not found. Provide a valid model_file path, place the file "
        f"at {cached_path}, or pass model_url to download it."
    )


@dataclass
class CGEntry:
    atom_name: str
    atom_type: int
    mass: float
    resname: str
    resid: int
    chain_id: str
    coord: np.ndarray


def _build_opeps_bonds(chain_residues: List[Dict]) -> List[Tuple[int, int]]:
    bonds: List[Tuple[int, int]] = []
    for chain in chain_residues:
        residues = chain["residues"]
        for idx, residue in enumerate(residues):
            resname = residue["resname"]
            beads = residue["beads"]
            if resname == "GLY":
                n_idx, ca_idx, c_idx, o_idx = beads
                bonds.extend([(n_idx, ca_idx), (ca_idx, c_idx), (c_idx, o_idx)])
            else:
                n_idx, ca_idx, cb_idx, c_idx, o_idx = beads
                bonds.extend(
                    [
                        (n_idx, ca_idx),
                        (ca_idx, cb_idx),
                        (ca_idx, c_idx),
                        (c_idx, o_idx),
                    ]
                )
            if idx < len(residues) - 1:
                next_n = residues[idx + 1]["beads"][0]
                bonds.append((beads[-2], next_n))
    return bonds


def _angles_from_bonds(bonds: Iterable[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    adjacency: Dict[int, set] = {}
    for i, j in bonds:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    angles = set()
    for center, neighbors in adjacency.items():
        neigh = sorted(neighbors)
        for i_idx in range(len(neigh)):
            for k_idx in range(i_idx + 1, len(neigh)):
                i = neigh[i_idx]
                k = neigh[k_idx]
                angles.add((i, center, k))
    return sorted(angles)


def _dihedrals_from_bonds(bonds: Iterable[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    adjacency: Dict[int, set] = {}
    for i, j in bonds:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    dihedrals = set()
    for j, neighbors_j in adjacency.items():
        for k in neighbors_j:
            neighbors_k = adjacency.get(k, set())
            for i in neighbors_j:
                if i == k:
                    continue
                for l in neighbors_k:
                    if l == j:
                        continue
                    dih = (i, j, k, l)
                    rev = (l, k, j, i)
                    if rev in dihedrals:
                        continue
                    dihedrals.add(dih)
    return sorted(dihedrals)


def build_cg_system_from_pdb(
    pdb_path: str,
    strip_hetero: bool = True,
) -> Tuple["mlcg.geometry.topology.Topology", np.ndarray, np.ndarray, np.ndarray, List[CGEntry], List[Dict]]:
    _ensure_mlcg("MLCG CG preparation")
    from mlcg.geometry.topology import Topology
    OPEPS_MAP = _import_opeps_map()

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())

    cg_entries: List[CGEntry] = []
    chain_residues: List[Dict] = []

    for chain in model:
        chain_entries = []
        for residue in chain:
            if strip_hetero and residue.id[0] != " ":
                continue
            resname = residue.get_resname().strip().upper()
            resname = _RESNAME_MAP.get(resname, resname)
            if resname not in _STANDARD_RESNAMES:
                raise ValueError(
                    f"Unsupported residue {resname} in {pdb_path}. "
                    "Only standard amino acids are supported."
                )
            required = ["N", "CA", "C", "O"]
            if resname != "GLY":
                required.append("CB")
            atoms = {}
            for atom_name in required:
                atom = _select_atom(residue, atom_name)
                if atom is None:
                    raise ValueError(
                        f"Missing heavy atom {atom_name} in residue {resname} "
                        f"{residue.id[1]} chain {chain.id} ({pdb_path})."
                    )
                atoms[atom_name] = atom

            if resname == "GLY":
                bead_atoms = ["N", "CA", "C", "O"]
            else:
                bead_atoms = ["N", "CA", "CB", "C", "O"]

            beads = []
            for atom_name in bead_atoms:
                key = (resname, atom_name)
                if key not in OPEPS_MAP:
                    raise ValueError(
                        f"No OPEPS mapping for {resname} {atom_name} in {pdb_path}."
                    )
                _, atom_type, mass = OPEPS_MAP[key]
                coord = np.array(atoms[atom_name].coord, dtype=float)
                cg_entries.append(
                    CGEntry(
                        atom_name=atom_name,
                        atom_type=int(atom_type),
                        mass=float(mass),
                        resname=resname,
                        resid=int(residue.id[1]),
                        chain_id=str(chain.id),
                        coord=coord,
                    )
                )
                beads.append(len(cg_entries) - 1)

            chain_entries.append(
                {"resname": resname, "resid": residue.id[1], "beads": beads}
            )
        if chain_entries:
            chain_residues.append({"chain_id": str(chain.id), "residues": chain_entries})

    if not cg_entries:
        raise ValueError(f"No protein residues found in {pdb_path}.")

    cg_topo = Topology()
    for entry in cg_entries:
        cg_topo.add_atom(
            entry.atom_type,
            entry.atom_name,
            entry.resname,
            entry.resid,
        )

    bonds = _build_opeps_bonds(chain_residues)
    for i, j in bonds:
        cg_topo.add_bond(i, j)

    angles = _angles_from_bonds(bonds)
    for i, j, k in angles:
        cg_topo.add_angle(i, j, k)

    dihedrals = _dihedrals_from_bonds(bonds)
    for i, j, k, l in dihedrals:
        cg_topo.add_dihedral(i, j, k, l)

    coords = np.stack([entry.coord for entry in cg_entries], axis=0)
    atom_types = np.array([entry.atom_type for entry in cg_entries], dtype=int)
    masses = np.array([entry.mass for entry in cg_entries], dtype=float)

    return cg_topo, coords, atom_types, masses, cg_entries, chain_residues


def _bead_map(residue: Dict) -> Dict[str, int]:
    beads = residue["beads"]
    if residue["resname"] == "GLY":
        return {"N": beads[0], "CA": beads[1], "C": beads[2], "O": beads[3]}
    return {
        "N": beads[0],
        "CA": beads[1],
        "CB": beads[2],
        "C": beads[3],
        "O": beads[4],
    }


def _terminal_atoms(chain_residues: List[Dict]) -> Tuple[set, set]:
    n_term_atoms: set = set()
    c_term_atoms: set = set()
    for chain in chain_residues:
        residues = chain["residues"]
        if len(residues) <= 1:
            continue
        n_term_atoms.update(_bead_map(residues[0]).values())
        c_term_atoms.update(_bead_map(residues[-1]).values())
    return n_term_atoms, c_term_atoms


def _atom_res_indices(chain_residues: List[Dict], n_atoms: int) -> np.ndarray:
    res_indices = np.zeros(n_atoms, dtype=int)
    res_counter = 0
    for chain in chain_residues:
        for residue in chain["residues"]:
            for atom_idx in _bead_map(residue).values():
                res_indices[atom_idx] = res_counter
            res_counter += 1
    return res_indices


def _split_bulk_termini(
    n_term_atoms: set, c_term_atoms: set, all_edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all_edges.size == 0:
        order = all_edges.shape[0] if all_edges.ndim == 2 else 2
        empty = np.zeros((order, 0), dtype=int)
        return empty, empty, empty
    edges_t = all_edges.T
    n_mask = np.isin(edges_t, list(n_term_atoms)).any(axis=1)
    c_mask = np.isin(edges_t, list(c_term_atoms)).any(axis=1)
    term_mask = n_mask | c_mask
    n_edges = edges_t[n_mask].T
    c_edges = edges_t[c_mask].T
    bulk_edges = edges_t[~term_mask].T
    return n_edges, c_edges, bulk_edges


def build_neighbor_list(
    cg_topo,
    chain_residues: List[Dict],
    cg_entries: Sequence[CGEntry],
    min_pair: int = 6,
    res_exclusion: int = 1,
) -> Dict[str, Dict]:
    _ensure_mlcg("MLCG neighbor list")
    from mlcg.geometry._symmetrize import _symmetrise_distance_interaction
    from mlcg.geometry.topology import get_connectivity_matrix, get_n_paths
    from mlcg.neighbor_list.neighbor_list import make_neighbor_list
    import networkx as nx
    from networkx.algorithms.shortest_paths.unweighted import bidirectional_shortest_path

    def check_graph_distance(
        graph: nx.Graph, conn_comp: List[set], node_1: int, node_2: int
    ) -> bool:
        con_1 = [i for i, comp in enumerate(conn_comp) if node_1 in comp][0]
        con_2 = [i for i, comp in enumerate(conn_comp) if node_2 in comp][0]
        if con_1 == con_2:
            shortest_path = bidirectional_shortest_path(graph, node_1, node_2)
            dist = len(shortest_path)
            return dist >= min_pair
        return True

    def list_to_edges(items: List[List[int]], order: int) -> np.ndarray:
        if not items:
            return np.zeros((order, 0), dtype=int)
        return np.array(items, dtype=int).T

    n_term_atoms, c_term_atoms = _terminal_atoms(chain_residues)
    atom_res_indices = _atom_res_indices(chain_residues, len(cg_entries))

    conn_mat = get_connectivity_matrix(cg_topo).numpy()
    bond_edges = get_n_paths(conn_mat, n=2).numpy()
    if bond_edges.size == 0:
        bond_edges = np.zeros((2, 0), dtype=int)
    angle_edges = get_n_paths(conn_mat, n=3).numpy()
    if angle_edges.size == 0:
        angle_edges = np.zeros((3, 0), dtype=int)

    n_term_bonds, c_term_bonds, bulk_bonds = _split_bulk_termini(
        n_term_atoms, c_term_atoms, bond_edges
    )
    n_term_angles, c_term_angles, bulk_angles = _split_bulk_termini(
        n_term_atoms, c_term_atoms, angle_edges
    )

    phi_edges = {res: [] for res in _ALL_RESIDUES}
    psi_edges = {res: [] for res in _ALL_RESIDUES}
    pro_omega: List[List[int]] = []
    non_pro_omega: List[List[int]] = []
    gamma_1: List[List[int]] = []
    gamma_2: List[List[int]] = []

    for chain in chain_residues:
        residues = chain["residues"]
        for idx, residue in enumerate(residues):
            resname = residue["resname"]
            bead_map = _bead_map(residue)
            if idx > 0:
                prev_map = _bead_map(residues[idx - 1])
                phi_edges[resname].append(
                    [prev_map["C"], bead_map["N"], bead_map["CA"], bead_map["C"]]
                )
                omega = [prev_map["CA"], prev_map["C"], bead_map["N"], bead_map["CA"]]
                if resname == "PRO":
                    pro_omega.append(omega)
                else:
                    non_pro_omega.append(omega)
            if idx < len(residues) - 1:
                next_map = _bead_map(residues[idx + 1])
                psi_edges[resname].append(
                    [bead_map["N"], bead_map["CA"], bead_map["C"], next_map["N"]]
                )
                gamma_2.append(
                    [bead_map["CA"], bead_map["O"], next_map["N"], bead_map["C"]]
                )
            if residue["resname"] != "GLY":
                gamma_1.append(
                    [bead_map["N"], bead_map["CB"], bead_map["C"], bead_map["CA"]]
                )

    fully_connected = _symmetrise_distance_interaction(
        cg_topo.fully_connected2torch()
    ).numpy()
    graph = nx.Graph(conn_mat)
    conn_comps = list(nx.connected_components(graph))
    pairs = []
    for p in fully_connected.T:
        i, j = int(p[0]), int(p[1])
        if abs(atom_res_indices[i] - atom_res_indices[j]) < res_exclusion:
            continue
        if graph.has_edge(i, j):
            continue
        if not check_graph_distance(graph, conn_comps, i, j):
            continue
        if bond_edges.size and np.all(bond_edges == p[:, None], axis=0).any():
            continue
        if angle_edges.size and np.all(angle_edges[[0, 2], :] == p[:, None], axis=0).any():
            continue
        pairs.append(p)
    if pairs:
        non_bonded_edges = torch.tensor(np.array(pairs).T, dtype=torch.long)
        non_bonded_edges = torch.unique(
            _symmetrise_distance_interaction(non_bonded_edges), dim=1
        ).numpy()
    else:
        non_bonded_edges = np.zeros((2, 0), dtype=int)

    edges_and_orders = [
        ("n_term_bonds", 2, n_term_bonds),
        ("bulk_bonds", 2, bulk_bonds),
        ("c_term_bonds", 2, c_term_bonds),
        ("n_term_angles", 3, n_term_angles),
        ("bulk_angles", 3, bulk_angles),
        ("c_term_angles", 3, c_term_angles),
        ("non_bonded", 2, non_bonded_edges),
        ("pro_omega", 4, list_to_edges(pro_omega, 4)),
        ("non_pro_omega", 4, list_to_edges(non_pro_omega, 4)),
        ("gamma_1", 4, list_to_edges(gamma_1, 4)),
        ("gamma_2", 4, list_to_edges(gamma_2, 4)),
    ]
    for res in _ALL_RESIDUES:
        edges_and_orders.append((f"{res}_phi", 4, list_to_edges(phi_edges[res], 4)))
        edges_and_orders.append((f"{res}_psi", 4, list_to_edges(psi_edges[res], 4)))

    nls: Dict[str, Dict] = {}
    for tag, order, edge in edges_and_orders:
        edge_t = torch.tensor(edge, dtype=torch.long) if not torch.is_tensor(edge) else edge
        nls[tag] = make_neighbor_list(tag, order, edge_t)
    return nls


def _write_cg_pdb(path: str, cg_entries: Sequence[CGEntry]) -> None:
    with open(path, "w") as handle:
        serial = 1
        for entry in cg_entries:
            atom_name = entry.atom_name.rjust(4)
            resname = entry.resname[:3].rjust(3)
            chain_id = (entry.chain_id or "A")[:1]
            resid = entry.resid % 10000
            x, y, z = entry.coord
            element = entry.atom_name[0].upper()
            line = (
                f"ATOM  {serial:5d} {atom_name} {resname} {chain_id}{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2}"
            )
            handle.write(line[:80].ljust(80) + "\n")
            serial += 1
        handle.write("END\n")


class mlcg_md:
    def __init__(self, input_pdb: str):
        _ensure_mlcg("mlcg_md")
        self.input_pdb = input_pdb
        self.pdb_name = os.path.basename(input_pdb).replace(".pdb", "")
        self.command_log: List[Dict[str, str]] = []
        self.cg_topology = None
        self.neighbor_list = None
        self.cg_entries: List[CGEntry] = []
        self.structure_file = None
        self.cg_pdb_file = None

    def prepare_inputs(
        self,
        output_dir: str,
        replicas: int = 1,
        coordinate_jitter: Optional[float] = None,
        strip_hetero: bool = True,
        write_cg_pdb: bool = True,
        structure_name: Optional[str] = None,
        cg_pdb_name: Optional[str] = None,
    ) -> List["mlcg.data.atomic_data.AtomicData"]:
        _ensure_mlcg("mlcg_md.prepare_inputs")
        from mlcg.data.atomic_data import AtomicData

        cg_topo, coords, atom_types, masses, cg_entries, chain_residues = build_cg_system_from_pdb(
            self.input_pdb,
            strip_hetero=strip_hetero,
        )
        neighbor_list = build_neighbor_list(cg_topo, chain_residues, cg_entries)

        os.makedirs(output_dir, exist_ok=True)
        if structure_name is None:
            structure_name = f"{self.pdb_name}_mlcg_structures.pt"
        structure_file = os.path.join(output_dir, structure_name)

        if cg_pdb_name is None:
            cg_pdb_name = f"{self.pdb_name}_cg.pdb"
        cg_pdb_file = os.path.join(output_dir, cg_pdb_name)

        base_pos = torch.tensor(coords, dtype=torch.float32)
        atom_types_t = torch.tensor(atom_types, dtype=torch.long)
        masses_t = torch.tensor(masses, dtype=torch.float32)

        data_list = []
        for idx in range(max(int(replicas), 1)):
            pos = base_pos.clone()
            if coordinate_jitter:
                noise = torch.randn_like(pos) * float(coordinate_jitter)
                pos = pos + noise
            data = AtomicData.from_points(
                pos=pos,
                atom_types=atom_types_t,
                masses=masses_t,
                neighborlist=copy.deepcopy(neighbor_list),
                tag=f"{self.pdb_name}_{idx + 1}",
            )
            data_list.append(data)

        torch.save(data_list, structure_file)
        if write_cg_pdb:
            _write_cg_pdb(cg_pdb_file, cg_entries)

        self.cg_topology = cg_topo
        self.neighbor_list = neighbor_list
        self.cg_entries = cg_entries
        self.structure_file = structure_file
        self.cg_pdb_file = cg_pdb_file

        return data_list


def write_mlcg_dcds(
    sims_dir: str,
    topology_pdb: str,
    output_dir: Optional[str] = None,
    pattern: str = "*_coords_*.npy",
    prefix: Optional[str] = None,
    stride: int = 1,
    overwrite: bool = False,
    validate_topology: bool = True,
) -> List[str]:
    """
    Convert MLCG numpy coordinate chunks into one DCD file per trajectory.

    Parameters
    ----------
    sims_dir : str
        Directory containing *_coords_*.npy files.
    topology_pdb : str
        CG PDB used as topology reference (for atom count validation).
    output_dir : str, optional
        Directory for output DCD files (default: sims_dir).
    pattern : str, optional
        Glob pattern to match coordinate chunks.
    prefix : str, optional
        Output file prefix. Defaults to the chunk filename prefix.
    stride : int, optional
        Keep every `stride`-th frame in each chunk.
    overwrite : bool, optional
        Overwrite existing DCDs if True.
    validate_topology : bool, optional
        Check that the PDB atom count matches the coordinate bead count.
    """

    def _coords_sort_key(path: str):
        match = re.search(r"_coords_(\d+)\.npy$", os.path.basename(path))
        if match:
            return int(match.group(1))
        return path

    if stride < 1:
        raise ValueError("stride must be >= 1.")

    coords_files = sorted(
        glob.glob(os.path.join(sims_dir, pattern)),
        key=_coords_sort_key,
    )
    if not coords_files:
        raise FileNotFoundError(f"No coordinate files found in {sims_dir} with {pattern}.")

    first = np.load(coords_files[0])
    if first.ndim != 4 or first.shape[-1] != 3:
        raise ValueError(
            "Expected coords with shape (n_traj, n_frames, n_beads, 3)."
        )
    n_traj, _, n_beads, _ = first.shape

    import mdtraj as md

    if validate_topology:
        top = md.load_pdb(topology_pdb).topology
        if top.n_atoms != n_beads:
            raise ValueError(
                f"Topology has {top.n_atoms} atoms but coordinates have {n_beads} beads."
            )

    if prefix is None:
        base = os.path.basename(coords_files[0])
        match = re.search(r"(.+)_coords_\d+\.npy$", base)
        prefix = match.group(1) if match else os.path.splitext(base)[0]

    output_dir = output_dir or sims_dir
    os.makedirs(output_dir, exist_ok=True)

    dcd_paths: List[str] = []
    writers = []
    try:
        for t_idx in range(n_traj):
            dcd_path = os.path.join(output_dir, f"{prefix}_traj{t_idx + 1:02d}.dcd")
            if os.path.exists(dcd_path) and not overwrite:
                raise FileExistsError(f"Refusing to overwrite {dcd_path}.")
            writers.append(md.formats.DCDTrajectoryFile(dcd_path, mode="w", force_overwrite=True))
            dcd_paths.append(dcd_path)

        for coords_path in coords_files:
            coords = np.load(coords_path)
            if coords.shape[0] != n_traj or coords.shape[2] != n_beads or coords.shape[3] != 3:
                raise ValueError(f"Unexpected shape for {coords_path}: {coords.shape}.")
            if stride > 1:
                coords = coords[:, ::stride, :, :]
            for t_idx, writer in enumerate(writers):
                writer.write(coords[t_idx])
    finally:
        for writer in writers:
            try:
                writer.close()
            except Exception:
                pass

    return dcd_paths
