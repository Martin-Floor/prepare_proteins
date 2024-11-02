import os
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from schrodinger import structure
from schrodinger.structutils import analyze

import argparse
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define input variables
parser = argparse.ArgumentParser(description='Analyze docking results.')
parser.add_argument('docking_folder', help='Path to the docking folder')
parser.add_argument('mae_output', help='Path to the Maestro output file')
parser.add_argument('model', help='Model name')
parser.add_argument('ligand', help='Ligand name')
parser.add_argument('--protein_atoms', default=None, help='JSON file including protein atoms for closest distance calculations.')
parser.add_argument('--atom_pairs', default=None, help='JSON file including atom pairs for distance calculations.')
parser.add_argument('--angles', default=None, help='JSON file including angles definitions.')
parser.add_argument('--return_failed', default=False, action='store_true', help='Output a file containing failed docking models.')
parser.add_argument('--ignore_hydrogens', default=False, action='store_true', help='Ignore hydrogens for closest distance calculation.')
parser.add_argument('--compute_sasa', action='store_true', help='Compute SASA values.')
parser.add_argument('--separator', default='-', help='Separator to use in naming model+ligand files.')
args = parser.parse_args()

# Assign arguments to variables
docking_folder = args.docking_folder
mae_output = args.mae_output
model = args.model
ligand = args.ligand

protein_atoms_file = args.protein_atoms
atom_pairs_file = args.atom_pairs
angles_file = args.angles
return_failed = args.return_failed
ignore_hydrogens = args.ignore_hydrogens
compute_sasa = args.compute_sasa
separator = args.separator

# Read input JSON files
protein_atoms = {}
atom_pairs = {}
angles = {}

if protein_atoms_file:
    with open(protein_atoms_file) as paf:
        protein_atoms = json.load(paf)
        logging.info(f'Read protein atoms from {protein_atoms_file}')

if atom_pairs_file:
    with open(atom_pairs_file) as apf:
        atom_pairs = json.load(apf)
        logging.info(f'Read atom pairs from {atom_pairs_file}')

if angles_file:
    with open(angles_file) as af:
        angles = json.load(af)
        logging.info(f'Read angles definitions from {angles_file}')

def RMSD(ref_coord, curr_coord):
    n_atoms = ref_coord.shape[0]
    if n_atoms == 0:
        raise ValueError("Reference coordinates are empty.")
    sq_distances = np.linalg.norm(ref_coord - curr_coord, axis=1)**2
    rmsd = np.sqrt(np.sum(sq_distances) / n_atoms)
    return rmsd

def computeLigandSASA(ligand_structure, protein_structure):
    ligand_atoms = [atom for atom in ligand_structure.atom]
    # Create a copy to avoid modifying the original ligand structure
    combined_structure = ligand_structure.copy()
    combined_structure.extend(protein_structure)
    sasa = analyze.calculate_sasa(combined_structure, atoms=ligand_atoms)
    logging.debug(f'Computed SASA: {sasa}')
    return sasa

def getAtomCoordinates(atoms, protein_coordinates, ligand_coordinates, ligand):
    """
    Retrieves coordinates and labels for specified atoms from protein and ligand structures.

    Parameters:
    - atoms: List of atoms (str for ligand atoms or list/tuple for protein/ligand atoms with full identifiers)
    - protein_coordinates: Dictionary mapping protein atom identifiers to coordinates
    - ligand_coordinates: Dictionary mapping ligand atom identifiers to coordinates
    - ligand: Ligand name for error messages

    Returns:
    - atoms_formatted: List of formatted atoms
    - coordinates: Dictionary mapping atoms to their coordinates
    - labels: Dictionary mapping atoms to their labels
    """
    atoms_formatted = []
    coordinates = {}
    labels = {}
    for atom in atoms:
        # Handle ligand atom specified by name (str)
        if isinstance(atom, str):
            found = False
            for key in ligand_coordinates:
                if key[-1] == atom:
                    coordinates[atom] = np.array(ligand_coordinates[key])
                    labels[atom] = atom  # Label as atom name
                    atoms_formatted.append(atom)
                    found = True
                    logging.debug(f'Ligand atom found: {atom} with label {labels[atom]}')
                    break
            if not found:
                logging.error(f'Atom {atom} was not found in ligand {ligand}.')
                raise ValueError(f'Atom {atom} was not found in ligand {ligand}.')
        # Handle protein or ligand atom specified by full identifier (list or tuple)
        elif isinstance(atom, (list, tuple)):
            atom = tuple(atom)
            if atom in protein_coordinates:
                coordinates[atom] = np.array(protein_coordinates[atom])
                # Format label without hyphens between chain and residue ID
                label = f"{atom[0]}{atom[1]}{atom[2]}"
                labels[atom] = label
                atoms_formatted.append(atom)
                logging.debug(f'Protein atom found: {atom} with label {labels[atom]}')
            elif atom in ligand_coordinates:
                coordinates[atom] = np.array(ligand_coordinates[atom])
                label = f"{atom[0]}{atom[1]}{atom[2]}"
                labels[atom] = label
                atoms_formatted.append(atom)
                logging.debug(f'Ligand atom found: {atom} with label {labels[atom]}')
            else:
                logging.error(f'Atom {atom} was not found in either protein or ligand.')
                raise ValueError(f'Atom {atom} was not found.')
        else:
            logging.error(f'Invalid atom specification: {atom}')
            raise ValueError(f'Atom specification {atom} is invalid.')
    return atoms_formatted, coordinates, labels

# Initialize data structures
data = {
    "Protein": [],
    "Ligand": [],
    "Pose": [],
    "Score": [],
    "RMSD": [],
}

if compute_sasa:
    data['SASA'] = []

if protein_atoms and model in protein_atoms and ligand in protein_atoms[model]:
    data["Closest distance"] = []
    data["Closest ligand atom"] = []
    data["Closest protein atom"] = []

if atom_pairs and model in atom_pairs and ligand in atom_pairs[model]:
    distance_data = {
        "Protein": [],
        "Ligand": [],
        "Pose": []
    }

if angles and model in angles and ligand in angles[model]:
    angle_data = {
        "Protein": [],
        "Ligand": [],
        "Pose": []
    }

logging.info(f'Processing model {model} and ligand {ligand}')

protein_structure = None
ligand_structures = []
pose_count = 0

# Read structures from Maestro file
for st in structure.StructureReader(mae_output):
    if 'r_i_glide_gscore' not in st.property:
        # Assume this is the protein structure
        protein_structure = st
        protein_coordinates = {}
        for atom in st.atom:
            if ignore_hydrogens and atom.element == 'H':
                continue
            residue = atom.getResidue()
            chain = residue.chain
            residue_id = residue.resnum
            atom_name = atom.pdbname.strip() or atom.name.strip()
            key = (chain, residue_id, atom_name)
            protein_coordinates[key] = atom.xyz
        logging.debug(f'Protein structure loaded with {len(protein_coordinates)} atoms.')
    else:
        # This is a ligand pose
        pose_count += 1
        ligand_structures.append((pose_count, st))
        logging.debug(f'Ligand pose {pose_count} loaded.')

if protein_structure is None:
    logging.error("Protein structure not found in mae_output.")
    raise ValueError("Protein structure not found in mae_output.")

ligand_coordinates = {}
scores = {}
element_counts = {}
if compute_sasa:
    sasa_values = {}

# Process ligand poses
for pose_number, st in ligand_structures:
    scores[pose_number] = st.property.get('r_i_glide_gscore', None)
    if compute_sasa:
        sasa_values[pose_number] = computeLigandSASA(st, protein_structure)
    ligand_coordinates[pose_number] = {}
    element_counts[pose_number] = {}
    for atom in st.atom:
        if ignore_hydrogens and atom.element == 'H':
            continue
        residue = atom.getResidue()
        chain = residue.chain
        residue_id = residue.resnum
        atom_name = atom.pdbname.strip() or atom.name.strip()
        if not atom_name:
            element_counts[pose_number].setdefault(atom.element, 0)
            element_counts[pose_number][atom.element] += 1
            atom_name = f"{atom.element}{element_counts[pose_number][atom.element]}"
        key = (chain, residue_id, atom_name)
        ligand_coordinates[pose_number][key] = atom.xyz
    if pose_number == 1:
        # Set reference coordinates for RMSD
        atom_keys = sorted(ligand_coordinates[pose_number].keys())
        reference_coordinates = np.array([ligand_coordinates[pose_number][a] for a in atom_keys])
        logging.debug(f'Reference coordinates set for pose {pose_number}.')

# Analyze docking poses
for pose in ligand_coordinates:
    # Store data
    data["Protein"].append(model)
    data["Ligand"].append(ligand)
    data["Pose"].append(pose)
    data["Score"].append(scores[pose])
    if compute_sasa:
        data["SASA"].append(sasa_values[pose])
    # Compute RMSD
    pose_coordinates = np.array([ligand_coordinates[pose][a] for a in atom_keys])
    rmsd = RMSD(reference_coordinates, pose_coordinates)
    data["RMSD"].append(rmsd)
    logging.debug(f'Pose {pose}: RMSD = {rmsd:.3f}')
    # Compute distances if atom_pairs is provided
    if atom_pairs and model in atom_pairs and ligand in atom_pairs[model]:
        # Store data
        distance_data["Protein"].append(model)
        distance_data["Ligand"].append(ligand)
        distance_data["Pose"].append(pose)
        for atoms in atom_pairs[model][ligand]:
            try:
                # Get coordinates
                atoms_formatted, coordinates, labels = getAtomCoordinates(atoms, protein_coordinates, ligand_coordinates[pose], ligand)
                # Compute distance
                distance = np.linalg.norm(coordinates[atoms_formatted[0]] - coordinates[atoms_formatted[1]])
                label = '-'.join([labels[a] for a in atoms_formatted])
                # Append distance
                distance_data.setdefault(label, []).append(distance)
                logging.debug(f'Distance between {labels[atoms_formatted[0]]} and {labels[atoms_formatted[1]]}: {distance:.3f}')
            except ValueError as ve:
                logging.error(str(ve))
                continue
    # Compute angles if angles is provided
    if angles and model in angles and ligand in angles[model]:
        # Store data
        angle_data["Protein"].append(model)
        angle_data["Ligand"].append(ligand)
        angle_data["Pose"].append(pose)
        for atoms in angles[model][ligand]:
            try:
                # Get coordinates
                atoms_formatted, coordinates, labels = getAtomCoordinates(atoms, protein_coordinates, ligand_coordinates[pose], ligand)
                # Compute angle
                v1 = coordinates[atoms_formatted[0]] - coordinates[atoms_formatted[1]]
                v2 = coordinates[atoms_formatted[2]] - coordinates[atoms_formatted[1]]
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                label = '-'.join([labels[a] for a in atoms_formatted])
                # Append angle
                angle_data.setdefault(label, []).append(angle)
                logging.debug(f'Angle at {labels[atoms_formatted[1]]} between {labels[atoms_formatted[0]]}, {labels[atoms_formatted[1]]}, {labels[atoms_formatted[2]]}: {angle:.2f}°')
            except ValueError as ve:
                logging.error(str(ve))
                continue
    # Compute closest distance if protein_atoms is provided
    if protein_atoms and model in protein_atoms and ligand in protein_atoms[model]:
        protein_atom_list = protein_atoms[model][ligand]
        try:
            # Get coordinates
            atoms_formatted, coordinates, labels = getAtomCoordinates(protein_atom_list, protein_coordinates, ligand_coordinates[pose], ligand)
            p_coordinates = np.array([coordinates[a] for a in atoms_formatted])
            l_coordinates = np.array([ligand_coordinates[pose][a] for a in atom_keys])
            ligand_atom_names = [a[-1] for a in atom_keys]
            protein_atom_names = [labels[a] for a in atoms_formatted]
            # Compute distance matrix
            M = distance_matrix(p_coordinates, l_coordinates)
            min_idx = np.unravel_index(np.argmin(M, axis=None), M.shape)
            min_distance = M[min_idx]
            data["Closest distance"].append(min_distance)
            data["Closest protein atom"].append(protein_atom_names[min_idx[0]])
            data["Closest ligand atom"].append(ligand_atom_names[min_idx[1]])
            logging.debug(f'Closest distance: {min_distance:.3f} between protein atom {protein_atom_names[min_idx[0]]} and ligand atom {ligand_atom_names[min_idx[1]]}')
        except ValueError as ve:
            logging.error(str(ve))
            data["Closest distance"].append(np.nan)
            data["Closest protein atom"].append(None)
            data["Closest ligand atom"].append(None)

# Create dataframes and save to CSV files
csv_name = f"{model}{separator}{ligand}.csv"

# Define output directories
scores_dir = os.path.join(docking_folder, '.analysis', 'scores')
atom_pairs_dir = os.path.join(docking_folder, '.analysis', 'atom_pairs')
angles_dir = os.path.join(docking_folder, '.analysis', 'angles')

# Ensure directories exist
os.makedirs(scores_dir, exist_ok=True)
os.makedirs(atom_pairs_dir, exist_ok=True)
os.makedirs(angles_dir, exist_ok=True)

# Save scores DataFrame
data_df = pd.DataFrame(data)
scores_csv = os.path.join(scores_dir, csv_name)
data_df.to_csv(scores_csv, index=False)
logging.info(f'Saved scores to {scores_csv}')

# Save distances DataFrame if applicable
if atom_pairs and model in atom_pairs and ligand in atom_pairs[model]:
    distance_df = pd.DataFrame(distance_data)
    distance_csv = os.path.join(atom_pairs_dir, csv_name)
    distance_df.to_csv(distance_csv, index=False)
    logging.info(f'Saved atom pair distances to {distance_csv}')

# Save angles DataFrame if applicable
if angles and model in angles and ligand in angles[model]:
    angle_df = pd.DataFrame(angle_data)
    angles_csv = os.path.join(angles_dir, csv_name)
    angle_df.to_csv(angles_csv, index=False)
    logging.info(f'Saved angles to {angles_csv}')

# If SASA was computed, it's already included in the scores DataFrame
if compute_sasa:
    logging.info('SASA computation completed and included in the scores DataFrame.')

logging.info('Finished processing model and ligand.')
