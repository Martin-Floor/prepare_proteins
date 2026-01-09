from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from Bio.Data.IUPACData import protein_letters_1to3
from Bio.PDB import NeighborSearch, PDBParser

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

Q_NATIVE_CUTOFF_A = 4.5
Q_MIN_SEQ_SEPARATION = 3
Q_LAMBDA = 1.5
Q_BETA_PER_A = 1.0  # 10 nm^-1 -> 1.0 A^-1
KBOLTZMANN = 0.0019872041


@dataclass
class MLCGRun:
    model_name: str
    protocol: str
    sims_dir: str
    prefix: str
    cg_pdb: Optional[str]
    coord_chunks: List[str]
    config_path: Optional[str] = None
    dcd_paths: List[str] = field(default_factory=list)
    n_traj: Optional[int] = None
    n_beads: Optional[int] = None
    expected_frames: Optional[int] = None


@dataclass
class NativeContactMap:
    ca_coords: np.ndarray
    ca_bead_indices: np.ndarray
    contact_pairs: np.ndarray
    contact_r0: np.ndarray
    residue_ids: List[Tuple[str, int]]


@dataclass
class NativeCAMap:
    ca_coords: np.ndarray
    ca_bead_indices: np.ndarray
    residue_ids: List[Tuple[str, int]]


def _normalize_chain_id(chain_id: str) -> str:
    return chain_id.strip() or "A"


def _is_heavy_atom(atom) -> bool:
    element = getattr(atom, "element", "").strip()
    if element:
        return element.upper() != "H"
    name = atom.get_name().strip().upper()
    return not name.startswith("H")


def _collect_native_residues(
    pdb_path: str,
    chain_ids: Optional[Iterable[str]] = None,
) -> List[Dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("native", pdb_path)
    model = next(structure.get_models())
    chain_filter = {c.strip() for c in chain_ids} if chain_ids else None

    chain_residues: List[Dict] = []
    for chain in model:
        chain_id = _normalize_chain_id(chain.id)
        if chain_filter and chain_id not in chain_filter:
            continue
        residues: List[Dict] = []
        for residue in chain:
            if residue.id[0] != " ":
                continue
            resname = residue.get_resname().strip().upper()
            resname = _RESNAME_MAP.get(resname, resname)
            if resname not in _STANDARD_RESNAMES:
                raise ValueError(
                    f"Unsupported residue {resname} in {pdb_path}. "
                    "Only standard amino acids are supported."
                )
            icode = residue.id[2]
            if str(icode).strip():
                raise ValueError(
                    f"Insertion codes are not supported (found {residue.id} in {pdb_path})."
                )
            ca_atom = residue["CA"] if "CA" in residue else None
            if ca_atom is None:
                raise ValueError(
                    f"Missing CA atom in residue {resname} {residue.id[1]} "
                    f"chain {chain_id} ({pdb_path})."
                )
            heavy_atoms = [atom for atom in residue if _is_heavy_atom(atom)]
            if not heavy_atoms:
                raise ValueError(
                    f"No heavy atoms found in residue {resname} {residue.id[1]} "
                    f"chain {chain_id} ({pdb_path})."
                )
            residues.append(
                {
                    "chain_id": chain_id,
                    "resid": int(residue.id[1]),
                    "resname": resname,
                    "ca_coord": np.array(ca_atom.coord, dtype=float),
                    "heavy_atoms": heavy_atoms,
                }
            )
        if residues:
            chain_residues.append({"chain_id": chain_id, "residues": residues})

    if not chain_residues:
        raise ValueError(f"No protein residues found in {pdb_path}.")
    return chain_residues


def _parse_cg_atom_indices(cg_pdb: str) -> Dict[Tuple[str, int, str], int]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("cg", cg_pdb)
    atom_indices: Dict[Tuple[str, int, str], int] = {}
    idx = 0
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        chain = residue.get_parent()
        chain_id = _normalize_chain_id(chain.id)
        icode = residue.id[2]
        if str(icode).strip():
            raise ValueError(
                f"Insertion codes are not supported in CG PDB (found {residue.id} in {cg_pdb})."
            )
        resid = int(residue.id[1])
        atom_name = atom.get_name().strip().upper()
        key = (chain_id, resid, atom_name)
        if key in atom_indices:
            raise ValueError(
                f"Duplicate atom entry for {key} in CG PDB {cg_pdb}."
            )
        atom_indices[key] = idx
        idx += 1
    return atom_indices


def _parse_cg_ca_coords(cg_pdb: str) -> Dict[Tuple[str, int], np.ndarray]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("cg", cg_pdb)
    ca_coords: Dict[Tuple[str, int], np.ndarray] = {}
    for atom in structure.get_atoms():
        if atom.get_name().strip().upper() != "CA":
            continue
        residue = atom.get_parent()
        chain = residue.get_parent()
        chain_id = _normalize_chain_id(chain.id)
        icode = residue.id[2]
        if str(icode).strip():
            raise ValueError(
                f"Insertion codes are not supported in CG PDB (found {residue.id} in {cg_pdb})."
            )
        resid = int(residue.id[1])
        key = (chain_id, resid)
        if key in ca_coords:
            raise ValueError(
                f"Duplicate CA entry for {chain_id}:{resid} in CG PDB {cg_pdb}."
            )
        ca_coords[key] = np.array(atom.coord, dtype=float)
    return ca_coords


def _native_contact_pairs(
    chain_residues: List[Dict],
    cutoff_a: float,
    min_seq_separation: int,
    include_interchain: bool = False,
) -> List[Tuple[str, int, int, float]]:
    if include_interchain:
        raise NotImplementedError(
            "Inter-chain native contacts are not implemented yet."
        )

    contacts: List[Tuple[str, int, int, float]] = []
    for chain in chain_residues:
        residues = chain["residues"]
        chain_id = chain["chain_id"]
        atom_list = []
        atom_to_res_idx: Dict[object, int] = {}
        for idx, residue in enumerate(residues):
            for atom in residue["heavy_atoms"]:
                atom_list.append(atom)
                atom_to_res_idx[atom] = idx

        if not atom_list:
            continue
        ns = NeighborSearch(atom_list)
        contact_pairs = set()
        for atom in atom_list:
            neighbors = ns.search(atom.coord, cutoff_a, level="A")
            for other in neighbors:
                if other is atom:
                    continue
                i = atom_to_res_idx[atom]
                j = atom_to_res_idx[other]
                if i == j:
                    continue
                if abs(i - j) < min_seq_separation:
                    continue
                pair = (i, j) if i < j else (j, i)
                contact_pairs.add(pair)

        for i, j in sorted(contact_pairs):
            res_i = residues[i]
            res_j = residues[j]
            r0 = float(
                np.linalg.norm(res_i["ca_coord"] - res_j["ca_coord"])
            )
            contacts.append((chain_id, res_i["resid"], res_j["resid"], r0))
    return contacts


def build_native_contact_map(
    native_pdb: str,
    cg_pdb: str,
    cutoff_a: float = Q_NATIVE_CUTOFF_A,
    min_seq_separation: int = Q_MIN_SEQ_SEPARATION,
    include_interchain: bool = False,
    r0_source: Literal["native", "cg"] = "native",
) -> NativeContactMap:
    """
    Build a native contact map. Contact pairs are defined from native_pdb, while
    r0 can be taken from the native structure or the CG model.
    """
    chain_residues = _collect_native_residues(native_pdb)
    contacts = _native_contact_pairs(
        chain_residues,
        cutoff_a=cutoff_a,
        min_seq_separation=min_seq_separation,
        include_interchain=include_interchain,
    )
    atom_indices = _parse_cg_atom_indices(cg_pdb)
    cg_ca_coords = None
    if r0_source not in ("native", "cg"):
        raise ValueError(f"Unsupported r0_source '{r0_source}'. Use 'native' or 'cg'.")
    if r0_source == "cg":
        cg_ca_coords = _parse_cg_ca_coords(cg_pdb)

    residue_ids: List[Tuple[str, int]] = []
    ca_coords: List[np.ndarray] = []
    ca_bead_indices: List[int] = []
    for chain in chain_residues:
        for residue in chain["residues"]:
            chain_id = residue["chain_id"]
            resid = residue["resid"]
            key = (chain_id, resid, "CA")
            if key not in atom_indices:
                raise ValueError(
                    f"Missing CA bead for {chain_id} residue {resid} in {cg_pdb}."
                )
            residue_ids.append((chain_id, resid))
            ca_coords.append(residue["ca_coord"])
            ca_bead_indices.append(atom_indices[key])

    contact_pairs = []
    contact_r0 = []
    for chain_id, resid_i, resid_j, r0 in contacts:
        key_i = (chain_id, resid_i, "CA")
        key_j = (chain_id, resid_j, "CA")
        if key_i not in atom_indices or key_j not in atom_indices:
            raise ValueError(
                f"Missing CA beads for contact {chain_id}:{resid_i}-{resid_j} in {cg_pdb}."
            )
        if r0_source == "cg":
            ca_key_i = (chain_id, resid_i)
            ca_key_j = (chain_id, resid_j)
            if cg_ca_coords is None:
                raise ValueError("CG CA coordinates were not loaded for r0_source='cg'.")
            if ca_key_i not in cg_ca_coords or ca_key_j not in cg_ca_coords:
                raise ValueError(
                    f"Missing CA coordinates for contact {chain_id}:{resid_i}-{resid_j} in {cg_pdb}."
                )
            r0 = float(
                np.linalg.norm(cg_ca_coords[ca_key_i] - cg_ca_coords[ca_key_j])
            )
        contact_pairs.append([atom_indices[key_i], atom_indices[key_j]])
        contact_r0.append(r0)

    if not contact_pairs:
        raise ValueError(
            f"No native contacts found in {native_pdb} using cutoff {cutoff_a} A."
        )

    return NativeContactMap(
        ca_coords=np.stack(ca_coords, axis=0),
        ca_bead_indices=np.array(ca_bead_indices, dtype=int),
        contact_pairs=np.array(contact_pairs, dtype=int),
        contact_r0=np.array(contact_r0, dtype=float),
        residue_ids=residue_ids,
    )


def build_native_ca_map(
    native_pdb: str,
    cg_pdb: str,
) -> NativeCAMap:
    chain_residues = _collect_native_residues(native_pdb)
    atom_indices = _parse_cg_atom_indices(cg_pdb)

    residue_ids: List[Tuple[str, int]] = []
    ca_coords: List[np.ndarray] = []
    ca_bead_indices: List[int] = []
    for chain in chain_residues:
        for residue in chain["residues"]:
            chain_id = residue["chain_id"]
            resid = residue["resid"]
            key = (chain_id, resid, "CA")
            if key not in atom_indices:
                raise ValueError(
                    f"Missing CA bead for {chain_id} residue {resid} in {cg_pdb}."
                )
            residue_ids.append((chain_id, resid))
            ca_coords.append(residue["ca_coord"])
            ca_bead_indices.append(atom_indices[key])

    return NativeCAMap(
        ca_coords=np.stack(ca_coords, axis=0),
        ca_bead_indices=np.array(ca_bead_indices, dtype=int),
        residue_ids=residue_ids,
    )


def _kabsch_rmsd(
    coords: np.ndarray,
    ref_coords: np.ndarray,
) -> np.ndarray:
    ref = ref_coords - ref_coords.mean(axis=0)
    n_atoms = ref.shape[0]
    rmsd = np.zeros(coords.shape[0], dtype=float)
    for i, frame in enumerate(coords):
        mobile = frame - frame.mean(axis=0)
        cov = mobile.T @ ref
        v, s, w = np.linalg.svd(cov)
        det = np.linalg.det(v @ w)
        if det < 0:
            v[:, -1] *= -1
        rot = v @ w
        aligned = mobile @ rot
        diff = aligned - ref
        rmsd[i] = np.sqrt(np.sum(diff * diff) / n_atoms)
    return rmsd


def _iter_coords_chunks(
    coord_paths: List[str],
    stride: int,
) -> Iterable[np.ndarray]:
    for coords_path in coord_paths:
        coords = np.load(coords_path)
        if coords.ndim != 4 or coords.shape[-1] != 3:
            raise ValueError(
                f"Unexpected coords shape for {coords_path}: {coords.shape}"
            )
        if stride > 1:
            coords = coords[:, ::stride, :, :]
        yield coords
class MLCGAnalysis:
    """
    Discover and analyze MLCG simulation outputs across a job folder.
    """

    def __init__(
        self,
        job_root: str,
        auto_convert: bool = False,
        stride: int = 1,
        dcd_output_root: Optional[str] = None,
        overwrite_mismatch: bool = True,
        validate_topology: bool = True,
    ) -> None:
        self.job_root = os.path.abspath(job_root)
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("stride must be >= 1.")
        self.dcd_output_root = (
            os.path.abspath(dcd_output_root)
            if dcd_output_root is not None
            else None
        )
        self.overwrite_mismatch = bool(overwrite_mismatch)
        self.validate_topology = bool(validate_topology)

        self.runs: List[MLCGRun] = []
        self.model_dirs: List[str] = []
        self._native_cache: Dict[Tuple[str, str, bool, str], NativeContactMap] = {}
        self._native_ca_cache: Dict[Tuple[str, str], NativeCAMap] = {}

        self._discover_runs()
        if auto_convert:
            self.ensure_dcds()

    def _discover_runs(self) -> None:
        self.model_dirs = self._find_model_dirs(self.job_root)
        for model_dir in self.model_dirs:
            model_name = os.path.basename(model_dir.rstrip(os.sep))
            input_dir = os.path.join(model_dir, "input_files")
            cg_pdb = self._find_cg_pdb(input_dir, model_name)
            for protocol_dir in self._find_protocol_dirs(model_dir):
                self.runs.extend(
                    self._collect_runs_from_protocol(
                        model_name, protocol_dir, cg_pdb
                    )
                )

    @staticmethod
    def _find_model_dirs(job_root: str) -> List[str]:
        if os.path.isdir(os.path.join(job_root, "input_files")):
            return [job_root]

        model_dirs: List[str] = []
        if not os.path.isdir(job_root):
            raise FileNotFoundError(f"Job folder not found: {job_root}")
        first_level: List[str] = []
        for entry in sorted(os.listdir(job_root)):
            path = os.path.join(job_root, entry)
            if not os.path.isdir(path):
                continue
            first_level.append(path)
            if os.path.isdir(os.path.join(path, "input_files")):
                model_dirs.append(path)
        if not model_dirs:
            for parent in first_level:
                for entry in sorted(os.listdir(parent)):
                    path = os.path.join(parent, entry)
                    if not os.path.isdir(path):
                        continue
                    if os.path.isdir(os.path.join(path, "input_files")):
                        model_dirs.append(path)
            if not model_dirs:
                raise FileNotFoundError(
                    f"No model folders with input_files found under {job_root}."
                )
        return model_dirs

    @staticmethod
    def _find_protocol_dirs(model_dir: str) -> List[str]:
        protocol_dirs: List[str] = []
        for entry in sorted(os.listdir(model_dir)):
            if entry.startswith(".") or entry == "input_files":
                continue
            path = os.path.join(model_dir, entry)
            if not os.path.isdir(path):
                continue
            sims_dir = os.path.join(path, "sims")
            if os.path.isdir(sims_dir):
                protocol_dirs.append(path)
                continue
            if glob.glob(os.path.join(path, "*_coords_*.npy")):
                protocol_dirs.append(path)
                continue
            if glob.glob(os.path.join(path, "*.json")):
                protocol_dirs.append(path)
        return protocol_dirs

    @staticmethod
    def _find_cg_pdb(input_dir: str, model_name: str) -> Optional[str]:
        if not os.path.isdir(input_dir):
            return None
        cg_candidates = sorted(glob.glob(os.path.join(input_dir, "*_cg.pdb")))
        if not cg_candidates:
            return None
        preferred = os.path.join(input_dir, f"{model_name}_cg.pdb")
        if preferred in cg_candidates:
            return preferred
        return cg_candidates[0]

    @staticmethod
    def _coords_sort_key(path: str):
        match = re.search(r"_coords_(\d+)\.npy$", os.path.basename(path))
        if match:
            return int(match.group(1))
        return path

    @staticmethod
    def _coords_prefix(path: str) -> Optional[str]:
        match = re.search(r"(.+)_coords_\d+\.npy$", os.path.basename(path))
        return match.group(1) if match else None

    @staticmethod
    def _load_config_prefixes(protocol_dir: str) -> Dict[str, str]:
        config_map: Dict[str, str] = {}
        for config_path in sorted(glob.glob(os.path.join(protocol_dir, "*.json"))):
            try:
                with open(config_path, "r") as handle:
                    config = json.load(handle)
            except Exception:
                continue
            sim_cfg = config.get("simulation", {})
            filename = sim_cfg.get("filename")
            if not filename:
                continue
            prefix = os.path.basename(filename)
            config_map[prefix] = config_path
        return config_map

    def _collect_runs_from_protocol(
        self,
        model_name: str,
        protocol_dir: str,
        cg_pdb: Optional[str],
    ) -> List[MLCGRun]:
        config_map = self._load_config_prefixes(protocol_dir)
        sims_dir = os.path.join(protocol_dir, "sims")
        search_dir = sims_dir if os.path.isdir(sims_dir) else protocol_dir
        coords_files = sorted(
            glob.glob(os.path.join(search_dir, "*_coords_*.npy")),
            key=self._coords_sort_key,
        )
        if not coords_files:
            return []

        grouped: Dict[str, List[str]] = {}
        for path in coords_files:
            prefix = self._coords_prefix(path)
            if not prefix:
                continue
            grouped.setdefault(prefix, []).append(path)

        runs: List[MLCGRun] = []
        protocol = os.path.basename(protocol_dir)
        for prefix, chunk_paths in sorted(grouped.items()):
            runs.append(
                MLCGRun(
                    model_name=model_name,
                    protocol=protocol,
                    sims_dir=os.path.dirname(chunk_paths[0]),
                    prefix=prefix,
                    cg_pdb=cg_pdb,
                    coord_chunks=sorted(chunk_paths, key=self._coords_sort_key),
                    config_path=config_map.get(prefix),
                )
            )
        return runs

    def _inspect_coords(self, run: MLCGRun, stride: int) -> Tuple[int, int, int]:
        n_traj = None
        n_beads = None
        total_frames = 0
        for coords_path in run.coord_chunks:
            coords = np.load(coords_path, mmap_mode="r")
            if coords.ndim != 4 or coords.shape[-1] != 3:
                raise ValueError(
                    f"Unexpected coords shape for {coords_path}: {coords.shape}"
                )
            if n_traj is None:
                n_traj = coords.shape[0]
                n_beads = coords.shape[2]
            else:
                if coords.shape[0] != n_traj or coords.shape[2] != n_beads:
                    raise ValueError(
                        f"Inconsistent coords shape for {coords_path}: {coords.shape}"
                    )
            frames = coords.shape[1]
            total_frames += (frames + stride - 1) // stride
        if n_traj is None or n_beads is None:
            raise ValueError(f"No coordinate data found for {run.prefix}.")
        return n_traj, n_beads, total_frames

    @staticmethod
    def _count_dcd_frames(dcd_path: str, topology: str) -> int:
        try:
            import mdtraj as md
        except Exception as exc:
            raise ImportError(
                "mdtraj is required to validate DCD frame counts."
            ) from exc
        try:
            dcd = md.formats.DCDTrajectoryFile(dcd_path, mode="r")
            try:
                return int(dcd.n_frames)
            except Exception:
                return int(len(dcd))
            finally:
                dcd.close()
        except Exception:
            try:
                traj = md.load(dcd_path, top=topology)
                return traj.n_frames
            except Exception as exc:
                raise RuntimeError(f"Failed to read DCD {dcd_path}.") from exc

    def _output_dir_for_run(self, run: MLCGRun) -> str:
        if self.dcd_output_root is None:
            return run.sims_dir
        return os.path.join(self.dcd_output_root, run.model_name, run.protocol)

    def ensure_dcds(
        self,
        stride: Optional[int] = None,
        overwrite_mismatch: Optional[bool] = None,
        force: bool = False,
    ) -> List[MLCGRun]:
        """
        Ensure DCD outputs exist and match expected frame counts.
        """
        stride = self.stride if stride is None else int(stride)
        if stride < 1:
            raise ValueError("stride must be >= 1.")
        overwrite_mismatch = (
            self.overwrite_mismatch
            if overwrite_mismatch is None
            else bool(overwrite_mismatch)
        )

        updated_runs: List[MLCGRun] = []
        for run in self.runs:
            if not run.coord_chunks:
                continue
            if not run.cg_pdb:
                raise FileNotFoundError(
                    f"CG PDB not found for {run.model_name} ({run.protocol})."
                )
            n_traj, n_beads, expected_frames = self._inspect_coords(run, stride)
            run.n_traj = n_traj
            run.n_beads = n_beads
            run.expected_frames = expected_frames

            output_dir = self._output_dir_for_run(run)
            dcd_paths = [
                os.path.join(output_dir, f"{run.prefix}_traj{idx + 1:02d}.dcd")
                for idx in range(n_traj)
            ]
            run.dcd_paths = dcd_paths

            needs_conversion = force
            if not needs_conversion:
                for dcd_path in dcd_paths:
                    if not os.path.exists(dcd_path):
                        needs_conversion = True
                        break
                    frames = self._count_dcd_frames(dcd_path, run.cg_pdb)
                    if frames != expected_frames:
                        needs_conversion = True
                        break

            if needs_conversion:
                from .mlcg_setup import write_mlcg_dcds

                os.makedirs(output_dir, exist_ok=True)
                run.dcd_paths = write_mlcg_dcds(
                    run.sims_dir,
                    run.cg_pdb,
                    output_dir=output_dir,
                    prefix=run.prefix,
                    stride=stride,
                    overwrite=overwrite_mismatch or force,
                    validate_topology=self.validate_topology,
                )
                updated_runs.append(run)

        return updated_runs

    def compute_rmsd(
        self,
        run: MLCGRun,
        native_pdb: Optional[str] = None,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute C-alpha RMSD to a native structure for each trajectory.

        If native_pdb is omitted and a native PDB named "<base>_native.pdb"
        exists in input_files (where <base> is derived from the CG PDB),
        it is used by default.
        """
        if not run.coord_chunks:
            raise ValueError("No coordinate chunks available for this run.")
        if native_pdb is None:
            native_pdb = self._guess_native_pdb(run)
            if not run.cg_pdb:
                raise ValueError("native_pdb is required when CG PDB is unavailable.")
            if native_pdb is None:
                native_pdb = run.cg_pdb
        if not run.cg_pdb:
            raise ValueError("CG PDB is required to map C-alpha beads.")

        stride = self.stride if stride is None else int(stride)
        if stride < 1:
            raise ValueError("stride must be >= 1.")

        native_ca = self._get_native_ca_map(native_pdb, run.cg_pdb)
        ca_indices = native_ca.ca_bead_indices
        ref_coords = native_ca.ca_coords

        n_traj = None
        checked = False
        rmsd_chunks: List[List[np.ndarray]] = []
        for coords in _iter_coords_chunks(run.coord_chunks, stride):
            if n_traj is None:
                n_traj = coords.shape[0]
                rmsd_chunks = [[] for _ in range(n_traj)]
            for t_idx in range(coords.shape[0]):
                traj = coords[t_idx]
                ca_coords = traj[:, ca_indices, :]
                if not checked:
                    if ca_coords.shape[1] != ref_coords.shape[0]:
                        raise ValueError(
                            "Native/Cg CA count mismatch for "
                            f"{run.model_name} ({run.protocol}). "
                            f"Native has {ref_coords.shape[0]} residues, "
                            f"trajectory has {ca_coords.shape[1]} CA beads. "
                            "Ensure native_pdb matches the CG model used for simulation."
                        )
                    checked = True
                rmsd_chunks[t_idx].append(
                    _kabsch_rmsd(ca_coords, ref_coords)
                )

        rmsd_trajs = [
            np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=float)
            for chunks in rmsd_chunks
        ]
        return np.stack(rmsd_trajs, axis=0)

    def compute_q(
        self,
        run: MLCGRun,
        native_pdb: Optional[str] = None,
        stride: Optional[int] = None,
        beta: float = Q_BETA_PER_A,
        lambda_val: float = Q_LAMBDA,
        include_interchain: bool = False,
        frame_chunk: int = 0,
        r0_source: Literal["native", "cg"] = "cg",
    ) -> np.ndarray:
        """
        Compute the fraction of native contacts Q for each trajectory.

        By default, native contacts are defined from native_pdb while r0 distances
        are taken from the CG model (r0_source="cg").
        """
        if not run.coord_chunks:
            raise ValueError("No coordinate chunks available for this run.")
        if not run.cg_pdb:
            raise ValueError("CG PDB is required to compute Q.")
        if native_pdb is None:
            native_pdb = self._guess_native_pdb(run)
            if native_pdb is None:
                raise FileNotFoundError(
                    "native_pdb is required when it cannot be inferred from input_files."
                )

        stride = self.stride if stride is None else int(stride)
        if stride < 1:
            raise ValueError("stride must be >= 1.")

        native_map = self._get_native_map(
            native_pdb,
            run.cg_pdb,
            include_interchain=include_interchain,
            r0_source=r0_source,
        )
        contact_pairs = native_map.contact_pairs
        r0 = native_map.contact_r0

        n_traj = None
        q_chunks: List[List[np.ndarray]] = []
        for coords in _iter_coords_chunks(run.coord_chunks, stride):
            if n_traj is None:
                n_traj = coords.shape[0]
                q_chunks = [[] for _ in range(n_traj)]
            for t_idx in range(coords.shape[0]):
                traj = coords[t_idx]
                q_vals = self._compute_q_for_traj(
                    traj,
                    contact_pairs,
                    r0,
                    beta=float(beta),
                    lambda_val=float(lambda_val),
                    frame_chunk=frame_chunk,
                )
                q_chunks[t_idx].append(q_vals)

        q_trajs = [
            np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=float)
            for chunks in q_chunks
        ]
        return np.stack(q_trajs, axis=0)

    def compute_fes_rmsd_q(
        self,
        model_name: Optional[str] = None,
        protocol: Optional[str] = None,
        prefix: Optional[str] = None,
        run_index: Optional[int] = None,
        native_pdb: Optional[str] = None,
        temperature: Optional[float] = None,
        beta_idx: Optional[int] = None,
        beta: Optional[float] = None,
        stride: Optional[int] = None,
        bins: Tuple[int, int] = (100, 100),
        rmsd_range: Optional[Tuple[float, float]] = None,
        q_range: Tuple[float, float] = (0.0, 1.0),
        q_beta: float = Q_BETA_PER_A,
        q_lambda: float = Q_LAMBDA,
        include_interchain: bool = False,
        frame_chunk: int = 0,
        r0_source: Literal["native", "cg"] = "cg",
    ) -> Dict[str, object]:
        """
        Compute a 2D free-energy surface (kT) over RMSD and Q for a selected temperature.

        The trajectory index is assumed to follow the beta ordering stored in the
        simulation config. If temperature/beta are omitted, the lowest temperature
        (largest beta) is used by default.

        RMSD values are reported in Angstroms.

        If multiple runs are present, use model_name/protocol/prefix or run_index
        to select the desired run.
        """
        run = self._select_run(
            model_name=model_name,
            protocol=protocol,
            prefix=prefix,
            run_index=run_index,
        )
        if native_pdb is None:
            native_pdb = self._guess_native_pdb(run)
            if native_pdb is None:
                raise FileNotFoundError(
                    "native_pdb is required when it cannot be inferred from input_files."
                )

        rmsd = self.compute_rmsd(run, native_pdb=native_pdb, stride=stride)
        q = self.compute_q(
            run,
            native_pdb=native_pdb,
            stride=stride,
            beta=q_beta,
            lambda_val=q_lambda,
            include_interchain=include_interchain,
            frame_chunk=frame_chunk,
            r0_source=r0_source,
        )

        if rmsd.shape != q.shape:
            raise ValueError(
                f"RMSD/Q shape mismatch for {run.prefix}: {rmsd.shape} vs {q.shape}"
            )

        betas = self._load_betas_for_run(run)
        selected_idx = self._select_beta_index(
            betas=betas,
            temperature=temperature,
            beta=beta,
            beta_idx=beta_idx,
        )
        if selected_idx is None:
            if rmsd.shape[0] == 1:
                selected_idx = 0
            else:
                raise ValueError(
                    "Unable to select a temperature: provide beta_idx, beta, or temperature."
                )
        if selected_idx < 0 or selected_idx >= rmsd.shape[0]:
            raise IndexError(
                f"beta_idx {selected_idx} is out of range for {rmsd.shape[0]} trajectories."
            )

        rmsd_vals = rmsd[selected_idx].ravel()
        q_vals = q[selected_idx].ravel()
        mask = np.isfinite(rmsd_vals) & np.isfinite(q_vals)
        if not np.any(mask):
            raise ValueError("No finite RMSD/Q values found for FES computation.")

        range_spec = [rmsd_range, q_range]
        counts, rmsd_edges, q_edges = np.histogram2d(
            rmsd_vals[mask],
            q_vals[mask],
            bins=bins,
            range=range_spec,
        )
        total = counts.sum()
        if total <= 0:
            raise ValueError("No samples found in RMSD/Q histogram.")

        prob = counts / total
        fes = np.full_like(prob, np.nan, dtype=float)
        populated = prob > 0
        fes[populated] = -np.log(prob[populated])
        fes -= np.nanmin(fes)

        rmsd_centers = 0.5 * (rmsd_edges[:-1] + rmsd_edges[1:])
        q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])

        selected_beta = None
        if betas and selected_idx is not None and selected_idx < len(betas):
            selected_beta = float(betas[selected_idx])

        return {
            "fes": fes,
            "counts": counts,
            "rmsd_edges": rmsd_edges,
            "q_edges": q_edges,
            "rmsd_centers": rmsd_centers,
            "q_centers": q_centers,
            "beta": selected_beta,
            "beta_idx": selected_idx,
        }

    def summarize_runs(
        self,
        model_name: Optional[str] = None,
        protocol: Optional[str] = None,
        prefix: Optional[str] = None,
        run_index: Optional[int] = None,
        stride: Optional[int] = None,
        print_summary: bool = True,
    ) -> List[Dict[str, object]]:
        """
        Summarize available simulation outputs.

        If multiple runs are present, use model_name/protocol/prefix or run_index
        to restrict the selection. Defaults to stride=1 to reflect saved frames.
        Time estimates assume dt is in ps and are reported in ns. When coordinate
        chunks are present, time is computed using save_interval (if available),
        otherwise export_interval.
        """
        if stride is None:
            stride = 1
        stride = int(stride)
        if stride < 1:
            raise ValueError("stride must be >= 1.")

        runs = self._filter_runs(
            model_name=model_name,
            protocol=protocol,
            prefix=prefix,
            run_index=run_index,
        )
        summaries: List[Dict[str, object]] = []
        for run in runs:
            n_traj = 0
            n_beads = 0
            total_frames = 0
            ca_beads = None
            if run.coord_chunks:
                n_traj, n_beads, total_frames = self._inspect_coords(run, stride)
                run.n_traj = n_traj
                run.n_beads = n_beads
                run.expected_frames = total_frames
            if run.cg_pdb and os.path.exists(run.cg_pdb):
                try:
                    ca_beads = len(_parse_cg_ca_coords(run.cg_pdb))
                except Exception:
                    ca_beads = None

            sim_settings = self._load_simulation_settings(run)
            dt = sim_settings.get("dt")
            export_interval = sim_settings.get("export_interval")
            save_interval = sim_settings.get("save_interval")
            n_timesteps = sim_settings.get("n_timesteps")

            frame_interval = save_interval or export_interval
            effective_interval = (
                int(frame_interval) * int(stride) if frame_interval is not None else None
            )
            time_per_frame = (
                float(dt) * float(effective_interval)
                if dt is not None and effective_interval is not None
                else None
            )
            time_per_traj = (
                total_frames * time_per_frame if time_per_frame is not None else None
            )
            total_time = (
                time_per_traj * n_traj if time_per_traj is not None else None
            )
            expected_frames = None
            expected_frames_strided = None
            expected_time_per_traj = None
            if n_timesteps is not None and frame_interval is not None:
                expected_frames = int(n_timesteps) // int(frame_interval)
                expected_frames_strided = (
                    (expected_frames + stride - 1) // stride
                    if expected_frames is not None
                    else None
                )
            if dt is not None and n_timesteps is not None:
                expected_time_per_traj = float(dt) * float(n_timesteps)
            time_per_frame_ns = (
                time_per_frame / 1000.0 if time_per_frame is not None else None
            )
            time_per_traj_ns = (
                time_per_traj / 1000.0 if time_per_traj is not None else None
            )
            total_time_ns = (
                total_time / 1000.0 if total_time is not None else None
            )
            expected_time_per_traj_ns = (
                expected_time_per_traj / 1000.0
                if expected_time_per_traj is not None
                else None
            )

            betas = self._load_betas_for_run(run)
            temps = (
                [1.0 / (KBOLTZMANN * b) for b in betas]
                if betas
                else []
            )

            summary = {
                "model_name": run.model_name,
                "protocol": run.protocol,
                "prefix": run.prefix,
                "sims_dir": run.sims_dir,
                "n_traj": n_traj,
                "n_beads": n_beads,
                "n_ca_beads": ca_beads,
                "frames_per_traj": total_frames,
                "stride": stride,
                "dt": dt,
                "export_interval": export_interval,
                "save_interval": save_interval,
                "n_timesteps": n_timesteps,
                "time_per_frame_ps": time_per_frame,
                "time_per_traj_ps": time_per_traj,
                "total_time_ps": total_time,
                "time_per_frame_ns": time_per_frame_ns,
                "time_per_traj_ns": time_per_traj_ns,
                "total_time_ns": total_time_ns,
                "frames_expected": expected_frames,
                "frames_expected_strided": expected_frames_strided,
                "time_per_traj_expected_ps": expected_time_per_traj,
                "time_per_traj_expected_ns": expected_time_per_traj_ns,
                "betas": betas,
                "temperatures": temps,
            }
            summaries.append(summary)

            if print_summary:
                header = f"{run.model_name} | {run.protocol} | {run.prefix}"
                print(header)
                print(f"  sims_dir: {run.sims_dir}")
                if run.coord_chunks:
                    frame_msg = (
                        f"  trajectories: {n_traj}, beads: {n_beads}, "
                        f"frames/trajectory: {total_frames} (stride={stride})"
                    )
                    if expected_frames_strided is not None:
                        frame_msg += f" of {expected_frames_strided}"
                    print(frame_msg)
                    if ca_beads is not None:
                        print(f"  CA beads: {ca_beads}")
                else:
                    print("  trajectories: 0 (no coordinate chunks found)")
                if dt is not None and export_interval is not None:
                    print(
                        f"  dt: {dt}, export_interval: {export_interval}, "
                        f"time/frame: {time_per_frame_ns} ns"
                    )
                    if save_interval is not None:
                        print(f"  save_interval: {save_interval}")
                    if time_per_traj_ns is not None:
                        print(f"  time/trajectory: {time_per_traj_ns} ns")
                    if expected_time_per_traj_ns is not None:
                        print(
                            f"  expected time/trajectory: {expected_time_per_traj_ns} ns"
                        )
                    if total_time_ns is not None:
                        print(f"  total time (all trajectories): {total_time_ns} ns")
                if temps:
                    temp_min = min(temps)
                    temp_max = max(temps)
                    print(
                        f"  temperatures (K): n={len(temps)}, "
                        f"min={temp_min:.2f}, max={temp_max:.2f}"
                    )
                else:
                    print("  temperatures (K): not found")
        return summaries

    @staticmethod
    def _compute_q_for_traj(
        traj: np.ndarray,
        contact_pairs: np.ndarray,
        r0: np.ndarray,
        beta: float,
        lambda_val: float,
        frame_chunk: int = 0,
    ) -> np.ndarray:
        if traj.ndim != 3:
            raise ValueError("Expected trajectory chunk with shape (n_frames, n_beads, 3).")
        n_frames = traj.shape[0]
        if frame_chunk and frame_chunk > 0:
            chunks = []
            for start in range(0, n_frames, frame_chunk):
                end = min(start + frame_chunk, n_frames)
                chunks.append(
                    MLCGAnalysis._compute_q_for_traj(
                        traj[start:end],
                        contact_pairs,
                        r0,
                        beta=beta,
                        lambda_val=lambda_val,
                        frame_chunk=0,
                    )
                )
            return np.concatenate(chunks, axis=0)

        i_idx = contact_pairs[:, 0]
        j_idx = contact_pairs[:, 1]
        delta = traj[:, i_idx, :] - traj[:, j_idx, :]
        dist = np.linalg.norm(delta, axis=-1)
        q = 1.0 / (1.0 + np.exp(beta * (dist - lambda_val * r0)))
        return np.mean(q, axis=1)

    def _get_native_map(
        self,
        native_pdb: str,
        cg_pdb: str,
        include_interchain: bool,
        r0_source: Literal["native", "cg"] = "native",
    ) -> NativeContactMap:
        key = (
            os.path.abspath(native_pdb),
            os.path.abspath(cg_pdb),
            include_interchain,
            r0_source,
        )
        native_map = self._native_cache.get(key)
        if native_map is None:
            native_map = build_native_contact_map(
                native_pdb,
                cg_pdb,
                include_interchain=include_interchain,
                r0_source=r0_source,
            )
            self._native_cache[key] = native_map
        return native_map

    @staticmethod
    def _parse_betas_from_yaml(config_path: str) -> List[float]:
        betas: List[float] = []
        in_betas = False
        with open(config_path, "r") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if not in_betas:
                    if stripped.startswith("betas:"):
                        in_betas = True
                    continue
                if stripped.startswith("-"):
                    token = stripped[1:].strip().split()[0]
                    try:
                        betas.append(float(token))
                    except ValueError:
                        break
                else:
                    if betas:
                        break
        return betas

    def _load_betas_for_run(self, run: MLCGRun) -> List[float]:
        betas: List[float] = []
        if run.config_path and os.path.exists(run.config_path):
            try:
                with open(run.config_path, "r") as handle:
                    config = json.load(handle)
                betas = config.get("betas", []) or []
            except Exception:
                betas = []
        if not betas and run.sims_dir and os.path.isdir(run.sims_dir):
            yaml_paths = sorted(glob.glob(os.path.join(run.sims_dir, "*_config.yaml")))
            for path in yaml_paths:
                betas = self._parse_betas_from_yaml(path)
                if betas:
                    break
        return [float(b) for b in betas] if betas else []

    @staticmethod
    def _parse_simulation_from_yaml(config_path: str) -> Dict[str, Optional[float]]:
        settings: Dict[str, Optional[float]] = {}
        in_sim = False
        targets = {"dt", "export_interval", "n_timesteps", "save_interval"}
        with open(config_path, "r") as handle:
            for line in handle:
                if not in_sim:
                    if line.strip().startswith("simulation:"):
                        in_sim = True
                    continue
                if not line.startswith((" ", "\t")):
                    if settings:
                        break
                    continue
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                key, sep, value = stripped.partition(":")
                if not sep:
                    continue
                key = key.strip()
                if key not in targets:
                    continue
                value = value.strip()
                if not value:
                    continue
                try:
                    parsed = float(value) if "." in value else int(value)
                except ValueError:
                    continue
                settings[key] = parsed
        return settings

    def _load_simulation_settings(self, run: MLCGRun) -> Dict[str, Optional[float]]:
        settings: Dict[str, Optional[float]] = {}
        if run.config_path and os.path.exists(run.config_path):
            try:
                with open(run.config_path, "r") as handle:
                    config = json.load(handle)
                sim_cfg = config.get("simulation", {})
                for key in ("dt", "export_interval", "n_timesteps", "save_interval"):
                    if key in sim_cfg:
                        settings[key] = sim_cfg[key]
            except Exception:
                settings = {}
        if not settings and run.sims_dir and os.path.isdir(run.sims_dir):
            yaml_paths = sorted(glob.glob(os.path.join(run.sims_dir, "*_config.yaml")))
            for path in yaml_paths:
                settings = self._parse_simulation_from_yaml(path)
                if settings:
                    break
        return settings

    def _select_beta_index(
        self,
        betas: List[float],
        temperature: Optional[float],
        beta: Optional[float],
        beta_idx: Optional[int],
    ) -> Optional[int]:
        if beta_idx is not None:
            return int(beta_idx)
        if not betas:
            return None
        if beta is None and temperature is None:
            return int(max(range(len(betas)), key=lambda i: betas[i]))
        if beta is None and temperature is not None:
            if temperature <= 0:
                raise ValueError("temperature must be positive.")
            beta = 1.0 / (KBOLTZMANN * float(temperature))
        target_beta = float(beta)
        return int(min(range(len(betas)), key=lambda i: abs(betas[i] - target_beta)))

    def _select_run(
        self,
        model_name: Optional[str],
        protocol: Optional[str],
        prefix: Optional[str],
        run_index: Optional[int],
    ) -> MLCGRun:
        if run_index is not None:
            if run_index < 0 or run_index >= len(self.runs):
                raise IndexError(f"run_index {run_index} is out of range.")
            return self.runs[run_index]

        candidates = self.runs
        if model_name is not None:
            candidates = [run for run in candidates if run.model_name == model_name]
        if protocol is not None:
            candidates = [run for run in candidates if run.protocol == protocol]
        if prefix is not None:
            candidates = [run for run in candidates if run.prefix == prefix]

        if not candidates:
            raise ValueError("No runs match the provided selection.")
        if len(candidates) > 1:
            run_list = ", ".join(
                f"{run.model_name}:{run.protocol}:{run.prefix}" for run in candidates
            )
            raise ValueError(
                "Multiple runs match; specify model_name, protocol, prefix, or run_index. "
                f"Matches: {run_list}"
            )
        return candidates[0]

    def _filter_runs(
        self,
        model_name: Optional[str],
        protocol: Optional[str],
        prefix: Optional[str],
        run_index: Optional[int],
    ) -> List[MLCGRun]:
        if run_index is not None:
            if run_index < 0 or run_index >= len(self.runs):
                raise IndexError(f"run_index {run_index} is out of range.")
            return [self.runs[run_index]]

        candidates = self.runs
        if model_name is not None:
            candidates = [run for run in candidates if run.model_name == model_name]
        if protocol is not None:
            candidates = [run for run in candidates if run.protocol == protocol]
        if prefix is not None:
            candidates = [run for run in candidates if run.prefix == prefix]
        if not candidates:
            raise ValueError("No runs match the provided selection.")
        return candidates

    def _guess_native_pdb(self, run: MLCGRun) -> Optional[str]:
        if not run.cg_pdb:
            return None
        input_dir = os.path.dirname(run.cg_pdb)
        cg_name = os.path.splitext(os.path.basename(run.cg_pdb))[0]
        base = cg_name[:-3] if cg_name.endswith("_cg") else cg_name
        candidate = os.path.join(input_dir, f"{base}_native.pdb")
        return candidate if os.path.exists(candidate) else None

    def _get_native_ca_map(
        self,
        native_pdb: str,
        cg_pdb: str,
    ) -> NativeCAMap:
        key = (os.path.abspath(native_pdb), os.path.abspath(cg_pdb))
        native_ca = self._native_ca_cache.get(key)
        if native_ca is None:
            native_ca = build_native_ca_map(native_pdb, cg_pdb)
            self._native_ca_cache[key] = native_ca
        return native_ca
