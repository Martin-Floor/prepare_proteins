try:
    import openmm as omm
    from openmm import unit as u
    from openmm.app import *
except Exception:  # pragma: no cover - fallback for older installs
    from simtk import openmm as omm
    from simtk import unit as u
    from simtk.openmm.app import *

import argparse
import json
import os
import re
import time

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for openmm_detachment.py") from exc


parser = argparse.ArgumentParser()
parser.add_argument("input_prmtop", help="Path to the AMBER prmtop file.")
parser.add_argument("input_coordinates", help="Path to the starting coordinates (PDB, inpcrd, or rst7).")
parser.add_argument("simulation_time", help="Total detachment simulation time in ns.")
parser.add_argument("--detachment_json", required=True, help="JSON file defining protein/polymer atom groups and the pull axis.")
parser.add_argument("--reference_inpcrd", help="Optional reference inpcrd/rst7 used to recover periodic box vectors when the coordinate file does not store them.")
parser.add_argument("--chunk_size", default=5.0, help="Chunk size in ns.")
parser.add_argument("--temperature", default=300.0, help="Temperature in K.")
parser.add_argument("--pressure", default=1.0, help="Pressure in bar for periodic systems.")
parser.add_argument("--collision_rate", default=1.0, help="Langevin collision rate in 1/ps.")
parser.add_argument("--time_step", default=0.002, help="Time step in ps.")
parser.add_argument("--dcd_report_time", default=10.0, help="Trajectory report interval in ps.")
parser.add_argument("--data_report_time", default=5.0, help="Thermodynamic/data report interval in ps.")
parser.add_argument("--pull_update_time", default=10.0, help="Interval in ps used to update the moving pull target.")
parser.add_argument("--nvt_time", type=float, default=0.0, help="Optional NVT equilibration time in ns.")
parser.add_argument("--npt_time", type=float, default=0.0, help="Optional NPT equilibration time in ns.")
parser.add_argument("--nvt_temp_scaling_steps", type=int, default=10, help="Temperature scaling segments during optional NVT equilibration.")
parser.add_argument("--npt_restraint_scaling_steps", type=int, default=10, help="Restraint scaling segments during optional NPT equilibration.")
parser.add_argument("--restraint_constant", type=float, default=10.0, help="Equilibration positional restraint constant in kcal/mol/A^2.")
parser.add_argument("--equilibration_data_report_time", type=float, default=1.0, help="Equilibration data report interval in ps.")
parser.add_argument("--equilibration_dcd_report_time", type=float, default=0.0, help="Equilibration DCD report interval in ps (0 disables reporting).")
parser.add_argument("--restraint_residue_pairs_json", help="Optional JSON file describing the heavy-atom restraint selection during equilibration.")
parser.add_argument("--spring_constant", type=float, default=0.5, help="Detachment spring constant in kcal/mol/A^2.")
parser.add_argument("--target_displacement_a", type=float, default=25.0, help="Total target displacement along the slab normal in Angstrom.")
parser.add_argument("--contact_cutoff_a", type=float, default=4.5, help="Heavy-atom cutoff in Angstrom used for protein-polymer contact reporting.")
parser.add_argument("--run_minimization", action="store_true", help="Run energy minimization before equilibration/production.")
parser.add_argument("--seed", type=int, help="Seed for initial velocities.")
args = parser.parse_args()


def _ns_to_steps(duration_ns, time_step_ps, label, allow_zero=False):
    if allow_zero and duration_ns == 0:
        return 0
    steps_float = (duration_ns * 1000.0) / time_step_ps
    if steps_float < 1:
        raise ValueError(
            f"{label} of {duration_ns} ns produces fewer than one integration step with a {time_step_ps} ps time step."
        )
    steps = int(round(steps_float))
    if steps <= 0:
        raise ValueError(f"{label} results in an invalid number of steps ({steps}).")
    return steps


def _interval_to_steps(interval_ps, time_step_ps, label, allow_zero=False):
    if allow_zero and interval_ps == 0:
        return None
    steps_float = interval_ps / time_step_ps
    if steps_float < 1:
        raise ValueError(f"{label} of {interval_ps} ps is shorter than the integration time step ({time_step_ps} ps).")
    steps = int(round(steps_float))
    if steps <= 0:
        raise ValueError(f"{label} results in an invalid number of steps ({steps}).")
    return steps


def _split_steps(total_steps, segments, label):
    if total_steps == 0:
        return []
    if segments < 1:
        raise ValueError(f"{label} must be at least 1.")
    if total_steps < segments:
        raise ValueError(
            f"{label} ({segments}) exceeds the available MD steps ({total_steps}). Reduce the number of segments or increase the simulation length."
        )
    base, remainder = divmod(total_steps, segments)
    return [base + (1 if i < remainder else 0) for i in range(segments)]


def _to_production_steps(step, equilibration_steps):
    if step is None:
        return None
    if step <= 0:
        return 0
    if step >= equilibration_steps:
        return step - equilibration_steps
    return step


def _normalize_chain_id(chain_id):
    if chain_id is None:
        return ""
    return str(chain_id).strip()


def _residue_pair_key(chain_id, residue_id):
    chain = _normalize_chain_id(chain_id)
    try:
        residue = int(str(residue_id).strip())
    except Exception as exc:
        raise ValueError(f"Invalid residue ID in restraint selection: {residue_id!r}") from exc
    return (chain, residue)


def _normalize_residue_id(residue_id):
    try:
        return int(str(residue_id).strip())
    except Exception as exc:
        raise ValueError(f"Invalid residue ID in restraint selection: {residue_id!r}") from exc


def _load_restraint_residue_pairs(json_path):
    if json_path is None:
        return None, None
    with open(json_path, "r") as handle:
        payload = json.load(handle)
    residue_pairs = None
    residue_ids = None
    if isinstance(payload, dict):
        if "residue_pairs" in payload:
            pairs = payload["residue_pairs"]
            if not isinstance(pairs, list):
                raise ValueError("'residue_pairs' must be a list.")
            residue_pairs = set()
            for entry in pairs:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError(f"Invalid restraint residue-pair entry: {entry!r}")
                residue_pairs.add(_residue_pair_key(entry[0], entry[1]))
        if "residue_ids" in payload:
            ids = payload["residue_ids"]
            if not isinstance(ids, list):
                raise ValueError("'residue_ids' must be a list.")
            residue_ids = {_normalize_residue_id(entry) for entry in ids}
        if residue_pairs is None and residue_ids is None:
            raise ValueError("Restraint selection JSON must contain 'residue_pairs' and/or 'residue_ids'.")
    elif isinstance(payload, list):
        residue_pairs = set()
        for entry in payload:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(f"Invalid restraint residue-pair entry: {entry!r}")
            residue_pairs.add(_residue_pair_key(entry[0], entry[1]))
    else:
        raise ValueError("Restraint selection JSON must be a list or a dict.")
    return residue_pairs, residue_ids


def getHeavyAtoms(topology, exclude_solvent=True, residue_pairs=None, residue_ids=None):
    heavy_atoms = []
    ion_names = {"NA", "K", "CL", "MG", "CA", "ZN", "LI", "RB", "SR", "BA", "CU"}
    for residue in topology.residues():
        try:
            is_water = Topology.isWater(residue)
        except Exception:
            is_water = residue.name.upper() in {"HOH", "WAT", "TIP3", "SOL"}
        if exclude_solvent and is_water:
            continue
        if residue.name.upper() in ion_names:
            continue
        residue_id = getattr(residue, "id", None)
        if residue_pairs is not None or residue_ids is not None:
            chain = getattr(getattr(residue, "chain", None), "id", None)
            try:
                pair = _residue_pair_key(chain, residue_id)
            except ValueError:
                continue
            residue_number = pair[1]
            selected = False
            if residue_pairs is not None and pair in residue_pairs:
                selected = True
            if residue_ids is not None and residue_number in residue_ids:
                selected = True
            if not selected:
                continue
        for atom in residue.atoms():
            element = getattr(atom, "element", None)
            symbol = getattr(element, "symbol", None)
            if symbol == "H":
                continue
            heavy_atoms.append(atom)
    return heavy_atoms


def addHeavyAtomsPositionalRestraintForce(system, positions, topology, restraint_constant, has_box, residue_pairs=None, residue_ids=None):
    heavy_atoms = getHeavyAtoms(topology, residue_pairs=residue_pairs, residue_ids=residue_ids)
    if not heavy_atoms:
        raise ValueError("No heavy atoms matched the requested positional-restraint selection.")
    dist_expr = "periodicdistance(x, y, z, x0, y0, z0)" if has_box else "sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    force = omm.CustomExternalForce(f"k*{dist_expr}^2")
    force_constant = (
        restraint_constant * u.kilocalories_per_mole / u.angstroms**2
    ).value_in_unit(u.kilojoules_per_mole / u.nanometer**2)
    force.addGlobalParameter("k", force_constant)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for atom in heavy_atoms:
        force.addParticle(atom.index, positions[atom.index].value_in_unit(u.nanometers))
    system.addForce(force)
    return force


def removeHeavyAtomPositionalRestraintForce(system):
    for index, force in enumerate(system.getForces()):
        if isinstance(force, omm.CustomExternalForce) and "k" in [force.getGlobalParameterName(j) for j in range(force.getNumGlobalParameters())]:
            system.removeForce(index)
            break


def _chunk_number_from_filename(filename):
    match = re.match(r"^production_(\d+)\.data$", filename)
    if not match:
        return None
    return int(match.group(1))


def get_last_step_from_report():
    last_step = 0
    last_chunk_file = None
    last_chunk_number = None
    for filename in os.listdir("."):
        if not (filename.startswith("production_") and filename.endswith(".data")):
            continue
        chunk_number = _chunk_number_from_filename(filename)
        if chunk_number is None:
            continue
        if last_chunk_number is None or chunk_number > last_chunk_number:
            last_chunk_number = chunk_number
            last_chunk_file = filename
    if last_chunk_file:
        with open(last_chunk_file, "r") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                last_step = int(line.split(",")[0])
    return last_step, last_chunk_file


def calculate_zfill(num_chunks):
    return max(2, len(str(num_chunks)))


def check_rename_needed(zfill_length):
    for filename in os.listdir("."):
        if filename.startswith("production_") and (
            filename.endswith(".data") or filename.endswith(".dcd") or filename.endswith(".chk")
        ):
            parts = filename.split("_")
            chunk_num = parts[1].split(".")[0]
            if len(chunk_num) != zfill_length:
                return True
    return False


def rename_chunk_files(zfill_length):
    for filename in os.listdir("."):
        if filename.startswith("production_") and (
            filename.endswith(".data") or filename.endswith(".dcd") or filename.endswith(".chk")
        ):
            parts = filename.split("_")
            chunk_num = parts[1].split(".")[0]
            new_chunk_num = chunk_num.zfill(zfill_length)
            new_filename = filename.replace(chunk_num, new_chunk_num)
            os.rename(filename, new_filename)


def _chunk_index_from_report_file(report_file):
    if not report_file:
        return None
    chunk_number = _chunk_number_from_filename(report_file)
    if chunk_number is None:
        return None
    return chunk_number - 1


def _get_checkpoint_step(simulation):
    try:
        state = simulation.context.getState(getStep=True)
    except Exception:
        return None
    try:
        return int(state.getStepCount())
    except Exception:
        return None


def current_time():
    return time.time()


def print_elapsed_time(start, end, description):
    elapsed = end - start
    print(f"{description} took {elapsed:.2f} seconds.")


def compute_ns_per_day(simulation_time_ns, elapsed_seconds):
    return (simulation_time_ns / elapsed_seconds) * 86400


def _load_coordinate_source(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npz":
        payload = np.load(path)
        if "positions_nm" not in payload:
            raise ValueError(f"NPZ coordinate file is missing 'positions_nm': {path}")
        positions_nm = np.asarray(payload["positions_nm"], dtype=float)
        positions = positions_nm * u.nanometer
        box_vectors = None
        if "box_vectors_nm" in payload and payload["box_vectors_nm"].size:
            box_array = np.asarray(payload["box_vectors_nm"], dtype=float)
            box_vectors = tuple(omm.Vec3(*vector) * u.nanometer for vector in box_array)
        return {
            "positions": positions,
            "box_vectors": box_vectors,
            "format": "npz",
        }
    if suffix == ".pdb":
        pdb = PDBFile(path)
        return {
            "positions": pdb.positions,
            "box_vectors": pdb.topology.getPeriodicBoxVectors(),
            "format": "pdb",
        }
    if suffix in {".inpcrd", ".rst7"}:
        inpcrd = AmberInpcrdFile(path)
        return {
            "positions": inpcrd.positions,
            "box_vectors": inpcrd.boxVectors,
            "format": "amber",
        }
    raise ValueError(f"Unsupported coordinate input format: {path}")


def _load_detachment_definition(path):
    with open(path, "r") as handle:
        payload = json.load(handle)
    required = ["protein_atom_indices", "partner_atom_indices", "pull_axis"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Detachment definition is missing required key: {key}")
    protein_atom_indices = [int(index) for index in payload["protein_atom_indices"]]
    partner_atom_indices = [int(index) for index in payload["partner_atom_indices"]]
    pull_axis = np.asarray(payload["pull_axis"], dtype=float)
    norm = np.linalg.norm(pull_axis)
    if norm <= 0:
        raise ValueError("pull_axis must define a non-zero vector.")
    payload["protein_atom_indices"] = protein_atom_indices
    payload["partner_atom_indices"] = partner_atom_indices
    payload["pull_axis"] = (pull_axis / norm).tolist()
    return payload


def _normalize_vector(vector, label):
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm <= 0:
        raise ValueError(f"{label} must define a non-zero vector.")
    return array / norm


def _build_in_plane_axes(axis, axis_major=None, axis_minor=None):
    normal = _normalize_vector(axis, "pull_axis")
    if axis_major is None:
        trial = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(trial, normal)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=float)
        axis_major_array = trial - np.dot(trial, normal) * normal
    else:
        axis_major_array = np.asarray(axis_major, dtype=float)
        axis_major_array = axis_major_array - np.dot(axis_major_array, normal) * normal
    axis_major_array = _normalize_vector(axis_major_array, "axis_major")

    if axis_minor is None:
        axis_minor_array = np.cross(normal, axis_major_array)
    else:
        axis_minor_array = np.asarray(axis_minor, dtype=float)
        axis_minor_array = axis_minor_array - np.dot(axis_minor_array, normal) * normal
        axis_minor_array = axis_minor_array - np.dot(axis_minor_array, axis_major_array) * axis_major_array
    axis_minor_array = _normalize_vector(axis_minor_array, "axis_minor")

    if np.dot(np.cross(axis_major_array, axis_minor_array), normal) < 0:
        axis_minor_array = -axis_minor_array

    return normal, axis_major_array, axis_minor_array


def _protein_heavy_indices_and_residue_map(topology, protein_atom_indices):
    protein_atom_set = {int(atom_index) for atom_index in protein_atom_indices}
    heavy_indices = []
    residue_map = {}
    residue_position = 0
    for residue in topology.residues():
        residue_heavy = []
        residue_contains_protein_atom = False
        for atom in residue.atoms():
            if atom.index not in protein_atom_set:
                continue
            residue_contains_protein_atom = True
            element = getattr(atom, "element", None)
            symbol = getattr(element, "symbol", None)
            if symbol == "H":
                continue
            residue_heavy.append(atom.index)
        if residue_contains_protein_atom:
            residue_position += 1
            for atom_index in residue_heavy:
                heavy_indices.append(atom_index)
                residue_map[atom_index] = residue_position
    return np.asarray(heavy_indices, dtype=int), residue_map


def _partner_heavy_indices(topology, partner_atom_indices):
    partner_atom_set = {int(atom_index) for atom_index in partner_atom_indices}
    heavy_indices = []
    for atom in topology.atoms():
        if atom.index not in partner_atom_set:
            continue
        element = getattr(atom, "element", None)
        symbol = getattr(element, "symbol", None)
        if symbol == "H":
            continue
        heavy_indices.append(atom.index)
    return np.asarray(heavy_indices, dtype=int)


def _strip_header_token(token):
    return token.strip().lstrip("#").strip().strip('"')


def _get_last_report_metadata(report_file):
    if not report_file or not os.path.exists(report_file):
        return {}
    headers = None
    values = None
    with open(report_file, "r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                headers = [_strip_header_token(token) for token in stripped.split(",")]
                continue
            values = [token.strip() for token in stripped.split(",")]
    if not headers or not values or len(headers) != len(values):
        return {}
    data = {}
    for header, value in zip(headers, values):
        try:
            data[header] = float(value)
        except ValueError:
            data[header] = value
    return data


def _group_weights_from_system(system, atom_indices):
    weights = []
    for atom_index in atom_indices:
        mass = system.getParticleMass(int(atom_index))
        try:
            weight = float(mass.value_in_unit(u.dalton))
        except Exception:
            weight = float(mass)
        weights.append(weight)
    return weights


def _weighted_group_center_nm(positions_nm, atom_indices, weights):
    selected = positions_nm[np.asarray(atom_indices, dtype=int), :]
    weights_array = np.asarray(weights, dtype=float)
    denom = weights_array.sum()
    if denom <= 0:
        raise ValueError("Group weights sum to zero.")
    return np.sum(selected * weights_array[:, None], axis=0) / denom


def _projection_nm(positions_nm, atom_indices_1, weights_1, atom_indices_2, weights_2, axis):
    center_1 = _weighted_group_center_nm(positions_nm, atom_indices_1, weights_1)
    center_2 = _weighted_group_center_nm(positions_nm, atom_indices_2, weights_2)
    return float(np.dot(center_1 - center_2, axis))


class DetachmentMonitor:
    def __init__(
        self,
        protein_atom_indices,
        protein_weights,
        partner_atom_indices,
        partner_weights,
        partner_heavy_atom_indices,
        protein_heavy_atom_indices,
        protein_heavy_residue_ids,
        axis,
        axis_major,
        axis_minor,
        spring_constant_kcal_mol_a2,
        contact_cutoff_a,
    ):
        self.protein_atom_indices = np.asarray(protein_atom_indices, dtype=int)
        self.protein_weights = np.asarray(protein_weights, dtype=float)
        self.partner_atom_indices = np.asarray(partner_atom_indices, dtype=int)
        self.partner_weights = np.asarray(partner_weights, dtype=float)
        self.partner_heavy_atom_indices = np.asarray(partner_heavy_atom_indices, dtype=int)
        self.protein_heavy_atom_indices = np.asarray(protein_heavy_atom_indices, dtype=int)
        self.protein_heavy_residue_ids = np.asarray(protein_heavy_residue_ids, dtype=int)
        self.axis = _normalize_vector(axis, "pull_axis")
        _, self.axis_major, self.axis_minor = _build_in_plane_axes(self.axis, axis_major=axis_major, axis_minor=axis_minor)
        self.spring_constant_kcal_mol_a2 = float(spring_constant_kcal_mol_a2)
        self.contact_cutoff_nm = float(contact_cutoff_a) / 10.0

    def _centers_and_delta(self, positions_nm):
        protein_center_nm = _weighted_group_center_nm(positions_nm, self.protein_atom_indices, self.protein_weights)
        partner_center_nm = _weighted_group_center_nm(positions_nm, self.partner_atom_indices, self.partner_weights)
        delta_nm = protein_center_nm - partner_center_nm
        return protein_center_nm, partner_center_nm, delta_nm

    def metrics_from_positions_nm(self, positions_nm, target_projection_nm):
        _, partner_center_nm, delta_nm = self._centers_and_delta(positions_nm)
        projection_nm = float(np.dot(delta_nm, self.axis))
        major_offset_nm = float(np.dot(delta_nm, self.axis_major))
        minor_offset_nm = float(np.dot(delta_nm, self.axis_minor))

        partner_heavy_coords_nm = positions_nm[self.partner_heavy_atom_indices, :]
        partner_relative_nm = partner_heavy_coords_nm - partner_center_nm
        partner_major_nm = np.dot(partner_relative_nm, self.axis_major)
        partner_minor_nm = np.dot(partner_relative_nm, self.axis_minor)
        min_edge_margin_nm = min(
            major_offset_nm - float(np.min(partner_major_nm)),
            float(np.max(partner_major_nm)) - major_offset_nm,
            minor_offset_nm - float(np.min(partner_minor_nm)),
            float(np.max(partner_minor_nm)) - minor_offset_nm,
        )

        protein_heavy_coords_nm = positions_nm[self.protein_heavy_atom_indices, :]
        partner_tree = cKDTree(partner_heavy_coords_nm)
        min_distances_nm, nearest_partner_indices = partner_tree.query(protein_heavy_coords_nm, k=1)
        min_protein_partner_distance_a = float(np.min(min_distances_nm)) * 10.0

        contact_lists = partner_tree.query_ball_point(protein_heavy_coords_nm, r=self.contact_cutoff_nm)
        contact_mask = np.asarray([len(indexes) > 0 for indexes in contact_lists], dtype=bool)
        contact_protein_atom_count = int(np.count_nonzero(contact_mask))
        if contact_protein_atom_count:
            contact_residue_count = int(np.unique(self.protein_heavy_residue_ids[contact_mask]).size)
            contact_partner_atom_count = int(len({int(index) for indexes in contact_lists for index in indexes}))
        else:
            contact_residue_count = 0
            contact_partner_atom_count = 0

        projection_a = projection_nm * 10.0
        target_projection_a = float(target_projection_nm) * 10.0
        pull_force_kcal_mol_a = -self.spring_constant_kcal_mol_a2 * (projection_a - target_projection_a)

        return {
            "projection_a": projection_a,
            "major_offset_a": major_offset_nm * 10.0,
            "minor_offset_a": minor_offset_nm * 10.0,
            "min_edge_margin_a": min_edge_margin_nm * 10.0,
            "min_protein_partner_distance_a": min_protein_partner_distance_a,
            "contact_residue_count": contact_residue_count,
            "contact_protein_atom_count": contact_protein_atom_count,
            "contact_partner_atom_count": contact_partner_atom_count,
            "pull_force_kcal_mol_a": pull_force_kcal_mol_a,
        }


class DetachmentSchedule:
    def __init__(self, initial_projection_nm, target_displacement_nm, total_steps):
        self.initial_projection_nm = float(initial_projection_nm)
        self.target_displacement_nm = float(target_displacement_nm)
        self.total_steps = int(total_steps)
        self.final_projection_nm = self.initial_projection_nm + self.target_displacement_nm

    def target_for_step(self, step):
        if self.total_steps <= 0:
            return self.final_projection_nm
        clamped = min(max(int(step), 0), self.total_steps)
        fraction = clamped / float(self.total_steps)
        return self.initial_projection_nm + fraction * self.target_displacement_nm


class DetachmentDataReporter(StateDataReporter):
    def __init__(
        self,
        file,
        reportInterval,
        simulation,
        detachment_force_group,
        protein_atom_indices,
        protein_weights,
        partner_atom_indices,
        partner_weights,
        monitor,
        target_getter,
        protocol_work_getter,
        append=False,
    ):
        super().__init__(
            file,
            reportInterval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            append=append,
        )
        self.simulation = simulation
        self.detachment_force_group = int(detachment_force_group)
        self.protein_atom_indices = np.asarray(protein_atom_indices, dtype=int)
        self.protein_weights = np.asarray(protein_weights, dtype=float)
        self.partner_atom_indices = np.asarray(partner_atom_indices, dtype=int)
        self.partner_weights = np.asarray(partner_weights, dtype=float)
        self.monitor = monitor
        self.target_getter = target_getter
        self.protocol_work_getter = protocol_work_getter

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        headers.extend(
            [
                "DetachmentBias_kcal_mol",
                "Projection_A",
                "TargetProjection_A",
                "ProtocolWork_kcal_mol",
                "PullForceAlongRC_kcal_mol_A",
                "MajorOffset_A",
                "MinorOffset_A",
                "MinEdgeMargin_A",
                "MinProteinPartnerDistance_A",
                "ContactResidueCount",
                "ContactProteinAtomCount",
                "ContactPartnerAtomCount",
            ]
        )
        return headers

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        positions_nm = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
        current_target_nm = float(self.target_getter())
        metrics = self.monitor.metrics_from_positions_nm(positions_nm, current_target_nm)
        bias_energy = simulation.context.getState(
            getEnergy=True,
            groups={self.detachment_force_group},
        ).getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
        values.append(bias_energy)
        values.append(metrics["projection_a"])
        values.append(current_target_nm * 10.0)
        values.append(float(self.protocol_work_getter()))
        values.append(metrics["pull_force_kcal_mol_a"])
        values.append(metrics["major_offset_a"])
        values.append(metrics["minor_offset_a"])
        values.append(metrics["min_edge_margin_a"])
        values.append(metrics["min_protein_partner_distance_a"])
        values.append(metrics["contact_residue_count"])
        values.append(metrics["contact_protein_atom_count"])
        values.append(metrics["contact_partner_atom_count"])
        return values


def _load_positions_and_box(prmtop, coordinate_path, reference_inpcrd=None):
    source = _load_coordinate_source(coordinate_path)
    positions = source["positions"]
    box_vectors = source["box_vectors"]
    if box_vectors is None and reference_inpcrd is not None:
        reference = _load_coordinate_source(reference_inpcrd)
        box_vectors = reference["box_vectors"]
    if box_vectors is None:
        try:
            box_vectors = prmtop.topology.getPeriodicBoxVectors()
        except Exception:
            box_vectors = None
    return positions, box_vectors


try:
    input_prmtop = args.input_prmtop
    input_coordinates = args.input_coordinates
    simulation_time = float(args.simulation_time)
    chunk_size = float(args.chunk_size)
    temperature = float(args.temperature)
    pressure = float(args.pressure)
    time_step = float(args.time_step)
    collision_rate = float(args.collision_rate)
    dcd_report_time = float(args.dcd_report_time)
    data_report_time = float(args.data_report_time)
    pull_update_time = float(args.pull_update_time)
    nvt_time = float(args.nvt_time)
    npt_time = float(args.npt_time)
    nvt_temp_scaling_steps = int(args.nvt_temp_scaling_steps)
    npt_restraint_scaling_steps = int(args.npt_restraint_scaling_steps)
    restraint_constant = float(args.restraint_constant)
    equilibration_data_report_time = float(args.equilibration_data_report_time)
    equilibration_dcd_report_time = float(args.equilibration_dcd_report_time)
    spring_constant = float(args.spring_constant)
    target_displacement_a = float(args.target_displacement_a)
    contact_cutoff_a = float(args.contact_cutoff_a)
    seed = args.seed
    restraint_residue_pairs_json = args.restraint_residue_pairs_json
    detachment_json = args.detachment_json
    reference_inpcrd = args.reference_inpcrd
    run_minimization = bool(args.run_minimization)

    if (
        simulation_time <= 0
        or chunk_size <= 0
        or temperature <= 0
        or pressure <= 0
        or time_step <= 0
        or collision_rate <= 0
        or spring_constant <= 0
        or target_displacement_a <= 0
        or contact_cutoff_a <= 0
        or dcd_report_time <= 0
        or data_report_time <= 0
        or pull_update_time <= 0
        or equilibration_data_report_time <= 0
        or equilibration_dcd_report_time < 0
        or nvt_time < 0
        or npt_time < 0
        or restraint_constant < 0
    ):
        raise ValueError("Simulation, pulling, temperature, pressure, contact cutoff, and reporting parameters must be positive (equilibration times can be zero).")
    if nvt_temp_scaling_steps < 1 or npt_restraint_scaling_steps < 1:
        raise ValueError("Equilibration scaling steps must be at least 1.")
except ValueError as exc:
    print(f"Parameter error: {exc}")
    raise SystemExit(1)

try:
    total_steps = _ns_to_steps(simulation_time, time_step, "Simulation time")
    chunk_steps = _ns_to_steps(chunk_size, time_step, "Chunk size")
    nvt_steps = _ns_to_steps(nvt_time, time_step, "NVT time", allow_zero=True)
    npt_steps = _ns_to_steps(npt_time, time_step, "NPT time", allow_zero=True)
    equilibration_steps = nvt_steps + npt_steps
    equilibration_data_report_steps = _interval_to_steps(equilibration_data_report_time, time_step, "Equilibration data report time")
    equilibration_dcd_report_steps = _interval_to_steps(
        equilibration_dcd_report_time,
        time_step,
        "Equilibration DCD report time",
        allow_zero=True,
    )
    production_data_report_steps = _interval_to_steps(data_report_time, time_step, "Data report time")
    production_dcd_report_steps = _interval_to_steps(dcd_report_time, time_step, "DCD report time")
    pull_update_steps = _interval_to_steps(pull_update_time, time_step, "Pull update time")
    nvt_step_schedule = _split_steps(nvt_steps, nvt_temp_scaling_steps, "NVT temperature scaling steps")
    npt_step_schedule = _split_steps(npt_steps, npt_restraint_scaling_steps, "NPT restraint scaling steps")
    restraint_residue_pairs, restraint_residue_ids = _load_restraint_residue_pairs(restraint_residue_pairs_json)
    detachment_definition = _load_detachment_definition(detachment_json)
except ValueError as exc:
    print(f"Parameter error: {exc}")
    raise SystemExit(1)


try:
    prmtop = AmberPrmtopFile(input_prmtop)
    positions, box_vectors = _load_positions_and_box(prmtop, input_coordinates, reference_inpcrd=reference_inpcrd)
    has_box = box_vectors is not None or (prmtop.topology.getUnitCellDimensions() is not None)

    print("Initializing detachment system from input files:")
    print(f"\tprmtop file: {input_prmtop}")
    print(f"\tcoordinate file: {input_coordinates}")
    if reference_inpcrd:
        print(f"\tbox reference: {reference_inpcrd}")
    print(f"System atoms: {prmtop.topology.getNumAtoms()}")
    print(f"Periodic box present?: {has_box}")

    nb_method = PME if has_box else NoCutoff
    nb_cutoff = 1 * u.nanometer if has_box else None
    system_kwargs = dict(nonbondedMethod=nb_method, constraints=HBonds, rigidWater=True)
    if nb_cutoff is not None:
        system_kwargs["nonbondedCutoff"] = nb_cutoff
    if has_box:
        system_kwargs["ewaldErrorTolerance"] = 1e-4
    system = prmtop.createSystem(**system_kwargs)
    system.addForce(omm.CMMotionRemover(100))

    protein_atom_indices = detachment_definition["protein_atom_indices"]
    partner_atom_indices = detachment_definition["partner_atom_indices"]
    axis = np.asarray(detachment_definition["pull_axis"], dtype=float)
    axis_major_input = detachment_definition.get("axis_major")
    axis_minor_input = detachment_definition.get("axis_minor")
    axis, axis_major, axis_minor = _build_in_plane_axes(axis, axis_major=axis_major_input, axis_minor=axis_minor_input)
    protein_weights = _group_weights_from_system(system, protein_atom_indices)
    partner_weights = _group_weights_from_system(system, partner_atom_indices)
    partner_heavy_atom_indices = detachment_definition.get("partner_heavy_atom_indices")
    if partner_heavy_atom_indices is None:
        partner_heavy_atom_indices = _partner_heavy_indices(prmtop.topology, partner_atom_indices).tolist()
    partner_heavy_atom_indices = np.asarray([int(index) for index in partner_heavy_atom_indices], dtype=int)
    protein_heavy_atom_indices, protein_residue_map = _protein_heavy_indices_and_residue_map(prmtop.topology, protein_atom_indices)
    protein_heavy_residue_ids = np.asarray([protein_residue_map[int(index)] for index in protein_heavy_atom_indices], dtype=int)

    positions_nm = positions.value_in_unit(u.nanometer)
    initial_projection_nm = detachment_definition.get("initial_projection_nm")
    if initial_projection_nm is None:
        initial_projection_nm = _projection_nm(
            positions_nm,
            protein_atom_indices,
            protein_weights,
            partner_atom_indices,
            partner_weights,
            axis,
        )
    else:
        initial_projection_nm = float(initial_projection_nm)

    detachment_force = omm.CustomCentroidBondForce(
        2,
        "0.5*k_det*(proj-r0)^2;"
        "proj=(x1-x2)*nx + (y1-y2)*ny + (z1-z2)*nz",
    )
    detachment_force.addGlobalParameter(
        "k_det",
        (
            spring_constant * u.kilocalories_per_mole / u.angstroms**2
        ).value_in_unit(u.kilojoules_per_mole / u.nanometer**2),
    )
    detachment_force.addGlobalParameter("r0", initial_projection_nm)
    detachment_force.addPerBondParameter("nx")
    detachment_force.addPerBondParameter("ny")
    detachment_force.addPerBondParameter("nz")
    protein_group = detachment_force.addGroup(protein_atom_indices, protein_weights)
    partner_group = detachment_force.addGroup(partner_atom_indices, partner_weights)
    detachment_force.addBond([protein_group, partner_group], axis.tolist())
    detachment_force_index = system.addForce(detachment_force)
    detachment_force_group = detachment_force_index + 1
    detachment_force.setForceGroup(detachment_force_group)

    integrator = omm.LangevinMiddleIntegrator(
        temperature * u.kelvin,
        collision_rate / u.picosecond,
        time_step * u.picoseconds,
    )
    if seed is not None:
        integrator.setRandomNumberSeed(seed)

    with open("system.xml", "w") as handle:
        handle.write(omm.XmlSerializer.serialize(system))
    with open("integrator.xml", "w") as handle:
        handle.write(omm.XmlSerializer.serialize(integrator))

    simulation = Simulation(prmtop.topology, system, integrator)
    simulation.context.setPositions(positions)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)

    checkpoint_file = "production.chk"

    num_chunks = total_steps // chunk_steps
    remaining_steps = total_steps % chunk_steps
    if remaining_steps > 0:
        num_chunks += 1

    zfill_length = calculate_zfill(num_chunks)
    if check_rename_needed(zfill_length):
        rename_chunk_files(zfill_length)

    last_report_step, last_report_file = get_last_step_from_report()
    last_report_metadata = _get_last_report_metadata(last_report_file)
    last_report_step = _to_production_steps(last_report_step, equilibration_steps)
    last_report_chunk = _chunk_index_from_report_file(last_report_file)
    if last_report_chunk is None and last_report_step > 0:
        last_report_chunk = last_report_step // chunk_steps
    if last_report_chunk is not None and last_report_step > 0:
        report_chunk_start = last_report_chunk * chunk_steps
        if last_report_step < report_chunk_start:
            last_report_step = report_chunk_start + last_report_step

    checkpoint_loaded = False
    checkpoint_path = None
    checkpoint_label = None
    loaded_chunk_index = None

    if last_report_chunk is not None:
        chunk_suffix = str(last_report_chunk + 1).zfill(zfill_length)
        chunk_checkpoint_file = f"production_{chunk_suffix}.chk"
        if os.path.exists(chunk_checkpoint_file):
            checkpoint_path = chunk_checkpoint_file
            checkpoint_label = "Chunk"
            loaded_chunk_index = last_report_chunk
        elif os.path.exists(checkpoint_file):
            checkpoint_path = checkpoint_file
            checkpoint_label = "General"
    elif os.path.exists(checkpoint_file):
        checkpoint_path = checkpoint_file
        checkpoint_label = "General"

    if checkpoint_path:
        print(f"{checkpoint_label} checkpoint file {checkpoint_path} found. Loading checkpoint.")
        simulation.loadCheckpoint(checkpoint_path)
        checkpoint_loaded = True

    if not checkpoint_loaded:
        for chunk in range(num_chunks - 1, -1, -1):
            chunk_suffix = str(chunk + 1).zfill(zfill_length)
            chunk_checkpoint_file = f"production_{chunk_suffix}.chk"
            if os.path.exists(chunk_checkpoint_file):
                print(f"Chunk checkpoint file {chunk_checkpoint_file} found. Loading checkpoint.")
                simulation.loadCheckpoint(chunk_checkpoint_file)
                checkpoint_loaded = True
                checkpoint_path = chunk_checkpoint_file
                last_report_chunk = chunk
                loaded_chunk_index = chunk
                break

    if checkpoint_loaded:
        checkpoint_step = _get_checkpoint_step(simulation)
        checkpoint_step = _to_production_steps(checkpoint_step, equilibration_steps)
        last_step = 0
        if checkpoint_path == checkpoint_file:
            if last_report_chunk is not None:
                checkpoint_chunk_start = last_report_chunk * chunk_steps
                checkpoint_chunk_end = min(checkpoint_chunk_start + chunk_steps, total_steps)
                if last_report_step >= checkpoint_chunk_end:
                    last_step = checkpoint_chunk_end
                elif last_report_step > 0:
                    last_step = checkpoint_chunk_start
                else:
                    last_step = checkpoint_chunk_start
            elif checkpoint_step is not None and checkpoint_step > 0:
                last_step = checkpoint_step
        else:
            if last_report_step > 0:
                last_step = last_report_step
            elif checkpoint_step is not None and checkpoint_step > 0:
                if loaded_chunk_index is not None:
                    checkpoint_chunk_start = loaded_chunk_index * chunk_steps
                    if checkpoint_step < checkpoint_chunk_start:
                        checkpoint_step = checkpoint_chunk_start + checkpoint_step
                last_step = checkpoint_step
            elif loaded_chunk_index is not None:
                last_step = loaded_chunk_index * chunk_steps
            elif last_report_chunk is not None:
                last_step = last_report_chunk * chunk_steps

        if last_step >= total_steps:
            print(f"Simulation has already run for {simulation_time} ns, increase the simulation time to continue.")
            raise SystemExit(0)

        chunk = last_step // chunk_steps
    else:
        if run_minimization:
            print("Applying energy minimization.")
            minimization_start = current_time()
            simulation.minimizeEnergy()
            minimization_end = current_time()
            print_elapsed_time(minimization_start, minimization_end, "Energy minimization")
            with open("minimized.pdb", "w") as handle:
                state = simulation.context.getState(getPositions=True)
                PDBFile.writeFile(prmtop.topology, state.getPositions(), handle)

        if has_box:
            barostat = omm.MonteCarloBarostat(pressure * u.bar, temperature * u.kelvin)
            system.addForce(barostat)
            simulation.context.reinitialize(preserveState=True)
            simulation.context.setPositions(positions)
            if box_vectors is not None:
                simulation.context.setPeriodicBoxVectors(*box_vectors)

        if nvt_steps or npt_steps:
            if restraint_constant > 0:
                print("\tAdding positional restraint force for optional equilibration")
                posres_force = addHeavyAtomsPositionalRestraintForce(
                    system,
                    positions,
                    prmtop.topology,
                    restraint_constant,
                    has_box,
                    residue_pairs=restraint_residue_pairs,
                    residue_ids=restraint_residue_ids,
                )
                simulation.context.reinitialize(preserveState=True)
                simulation.context.setPositions(positions)
                if box_vectors is not None:
                    simulation.context.setPeriodicBoxVectors(*box_vectors)
            else:
                posres_force = None

            equilibration_reporter = StateDataReporter(
                "equilibration.data",
                equilibration_data_report_steps,
                step=True,
                potentialEnergy=True,
                temperature=True,
            )
            simulation.reporters.append(equilibration_reporter)
            if equilibration_dcd_report_steps:
                simulation.reporters.append(DCDReporter("equilibration.dcd", equilibration_dcd_report_steps))

            if nvt_steps:
                initial_temperature = 5 * u.kelvin
                simulation.context.setVelocitiesToTemperature(initial_temperature)
                print("Optional NVT equilibration:")
                print(f"\tSteps: {nvt_steps}")
                nvt_start = current_time()
                T_initial = 5.0
                T_final = temperature
                if nvt_temp_scaling_steps == 1:
                    temperature_schedule = [T_final * u.kelvin]
                else:
                    dT = (T_final - T_initial) / (nvt_temp_scaling_steps - 1)
                    temperature_schedule = [(T_initial + i * dT) * u.kelvin for i in range(nvt_temp_scaling_steps)]
                for segment_steps, current_temperature in zip(nvt_step_schedule, temperature_schedule):
                    integrator.setTemperature(current_temperature)
                    simulation.step(segment_steps)
                nvt_end = current_time()
                print_elapsed_time(nvt_start, nvt_end, "NVT equilibration")

            if npt_steps:
                simulation.context.setVelocitiesToTemperature(temperature * u.kelvin)
                print("Optional NPT equilibration:")
                print(f"\tSteps: {npt_steps}")
                npt_start = current_time()
                if posres_force is not None:
                    initial_restraint_constant = restraint_constant
                    denom = max(npt_restraint_scaling_steps - 1, 1)
                    for index, segment_steps in enumerate(npt_step_schedule):
                        t = index / denom
                        K = (
                            initial_restraint_constant * (1 - t) ** 4 * u.kilocalories_per_mole / u.angstroms**2
                        ).value_in_unit(u.kilojoules_per_mole / u.nanometer**2)
                        posres_force.setGlobalParameterDefaultValue(0, K)
                        simulation.context.setParameter("k", K)
                        simulation.step(segment_steps)
                else:
                    for segment_steps in npt_step_schedule:
                        simulation.step(segment_steps)
                npt_end = current_time()
                print_elapsed_time(npt_start, npt_end, "NPT equilibration")

            if posres_force is not None:
                removeHeavyAtomPositionalRestraintForce(system)
                simulation.context.reinitialize(preserveState=True)

            simulation.reporters = []

        if nvt_steps == 0 and npt_steps == 0:
            if seed is None:
                simulation.context.setVelocitiesToTemperature(temperature * u.kelvin)
            else:
                simulation.context.setVelocitiesToTemperature(temperature * u.kelvin, int(seed))

        last_step = 0
        chunk = 0

    schedule = DetachmentSchedule(
        initial_projection_nm=initial_projection_nm,
        target_displacement_nm=target_displacement_a / 10.0,
        total_steps=total_steps,
    )
    monitor = DetachmentMonitor(
        protein_atom_indices=protein_atom_indices,
        protein_weights=protein_weights,
        partner_atom_indices=partner_atom_indices,
        partner_weights=partner_weights,
        partner_heavy_atom_indices=partner_heavy_atom_indices,
        protein_heavy_atom_indices=protein_heavy_atom_indices,
        protein_heavy_residue_ids=protein_heavy_residue_ids,
        axis=axis,
        axis_major=axis_major,
        axis_minor=axis_minor,
        spring_constant_kcal_mol_a2=spring_constant,
        contact_cutoff_a=contact_cutoff_a,
    )
    current_target = {"value": schedule.target_for_step(last_step)}
    protocol_work = {"value": 0.0}
    resume_step = last_step
    resume_report_step = last_report_step
    resume_from_report = resume_step == resume_report_step and resume_step > 0
    if resume_from_report:
        if "TargetProjection_A" in last_report_metadata:
            current_target["value"] = float(last_report_metadata["TargetProjection_A"]) / 10.0
        if "ProtocolWork_kcal_mol" in last_report_metadata:
            protocol_work["value"] = float(last_report_metadata["ProtocolWork_kcal_mol"])
    simulation.context.setParameter("r0", current_target["value"])

    for chunk in range(chunk, num_chunks):
        chunk_start_step = chunk * chunk_steps
        chunk_end_step = min(chunk_start_step + chunk_steps, total_steps)
        chunk_suffix = str(chunk + 1).zfill(zfill_length)

        if last_step > chunk_start_step:
            print(f"Resuming from step {last_step}.")

        steps_to_run = chunk_end_step - max(last_step, chunk_start_step)
        if steps_to_run <= 0:
            print("\tChunk already complete; skipping.")
            continue

        data_file = f"production_{chunk_suffix}.data"
        dcd_file = f"production_{chunk_suffix}.dcd"
        data_file_exists = os.path.isfile(data_file)
        dcd_file_exists = os.path.isfile(dcd_file)
        append_data = data_file_exists and resume_from_report and (chunk_start_step < resume_step <= chunk_end_step)
        append_dcd = dcd_file_exists and append_data

        try:
            production_reporter = DetachmentDataReporter(
                data_file,
                production_data_report_steps,
                simulation,
                detachment_force_group,
                protein_atom_indices,
                protein_weights,
                partner_atom_indices,
                partner_weights,
                monitor,
                lambda: current_target["value"],
                lambda: protocol_work["value"],
                append=append_data,
            )
        except TypeError:
            data_mode = "a" if append_data else "w"
            production_reporter = DetachmentDataReporter(
                open(data_file, data_mode),
                production_data_report_steps,
                simulation,
                detachment_force_group,
                protein_atom_indices,
                protein_weights,
                partner_atom_indices,
                partner_weights,
                monitor,
                lambda: current_target["value"],
                lambda: protocol_work["value"],
                append=False,
            )

        production_dcd_reporter = DCDReporter(dcd_file, production_dcd_report_steps, append=append_dcd)
        production_chk_reporter = CheckpointReporter(f"production_{chunk_suffix}.chk", production_data_report_steps)
        simulation.reporters = [production_reporter, production_dcd_reporter, production_chk_reporter]
        simulation.context.reinitialize(preserveState=True)

        print(f"Running detachment chunk {chunk_suffix} from step {chunk_start_step} to {chunk_end_step}")
        chunk_start_time = current_time()

        while last_step < chunk_end_step:
            segment_end = min(last_step + pull_update_steps, chunk_end_step)
            next_target = schedule.target_for_step(segment_end)
            if abs(next_target - current_target["value"]) > 1e-12:
                current_positions_nm = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
                current_metrics = monitor.metrics_from_positions_nm(current_positions_nm, current_target["value"])
                projection_delta_a = current_metrics["projection_a"] - (next_target * 10.0)
                previous_delta_a = current_metrics["projection_a"] - (current_target["value"] * 10.0)
                protocol_work["value"] += 0.5 * spring_constant * (
                    projection_delta_a ** 2 - previous_delta_a ** 2
                )
                current_target["value"] = next_target
                simulation.context.setParameter("r0", current_target["value"])
            segment_steps = segment_end - last_step
            simulation.step(segment_steps)
            last_step = segment_end

        simulation.saveCheckpoint(checkpoint_file)

        chunk_end_time = current_time()
        chunk_elapsed = chunk_end_time - chunk_start_time
        actual_ns = steps_to_run * time_step / 1000.0
        ns_per_day = compute_ns_per_day(actual_ns, chunk_elapsed)
        print_elapsed_time(chunk_start_time, chunk_end_time, f"Detachment chunk {chunk_suffix}")
        print(f"Detachment chunk {chunk_suffix} performance: {ns_per_day:.2f} ns/day")

    print("Detachment run completed.")

    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open("final_state.pdb", "w") as handle:
        PDBFile.writeFile(simulation.topology, state.getPositions(), handle)
    with open("final_state.xml", "w") as handle:
        handle.write(omm.XmlSerializer.serialize(state))

except Exception as exc:
    print(f"An error occurred: {exc}")
    raise
