import os
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from schrodinger import structure
from schrodinger.structutils import analyze

import argparse
import json

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('docking_folder', default=None, help='Path to the docking folder')
parser.add_argument('mae_output', default=None, help='Path to the docking folder')
parser.add_argument('model', default=None, help='Model name')
parser.add_argument('ligand', default=None, help='Ligand name')
parser.add_argument('--protein_atoms', default=None, help='Dictionary json file including protein_atoms for closest distance calculations.')
parser.add_argument('--atom_pairs', default=None, help='Dictionary json file including atom pairs for distance calculations.')
parser.add_argument('--angles', default=None, help='Dictionary json file including angles definitions.')
parser.add_argument('--return_failed', default=False, action='store_true', help='Output a file containing failed docking models.')
parser.add_argument('--ignore_hydrogens', default=False, action='store_true', help='Ignore hydrogens for closes distance calculation.')
parser.add_argument('--separator', default='-', help='Separator to use in naming model+ligand files.')
args=parser.parse_args()

docking_folder = args.docking_folder
mae_output = args.mae_output
model = args.model
ligand = args.ligand

protein_atoms = args.protein_atoms
atom_pairs = args.atom_pairs
angles = args.angles
return_failed = args.return_failed
ignore_hydrogens = args.ignore_hydrogens
separator = args.separator

def RMSD(ref_coord, curr_coord):
    sq_distances = np.linalg.norm(ref_coord - curr_coord, axis=1)**2
    rmsd = np.sqrt(np.sum(sq_distances)/ref_coord.shape[0])
    return rmsd

def computeLigandSASA(ligand_structure, protein_structure):

    ligand_atoms = []
    for atom in ligand_structure.atom:
        ligand_atoms.append(atom)
    ligand_structure.extend(protein_structure)
    return analyze.calculate_sasa(ligand_structure, atoms=ligand_atoms)

def getAtomCoordinates(atoms, protein_coordinates, ligand_coordinates):

    atoms_formated = []
    coordinates = {}
    labels = {}
    for atom in atoms:
        # Get as ligand coordinate if given as a single string
        if isinstance(atom, str):
            coords = {a[-1]:ligand_coordinates[a] for a in ligand_coordinates}
            if atom in coords:
                 coordinates[atom] = np.array(coords[atom])
            else:
                raise ValueError(f'Atom {atom} was not found in ligand {ligand}.')
            labels[atom] = atom

        # Get as protein or ligand coordinate if given as a list or tuple
        elif isinstance(atom, list):
            atom = tuple(atom)
            # Check if it's a protein atom
            if atom in protein_coordinates:
                coordinates[atom] = np.array(protein_coordinates[atom])
            # Check if it's a ligand atom
            elif atom in ligand_coordinates:
                coordinates[atom] = np.array(ligand_coordinates[atom])
            else:
                raise ValueError(f'Atom {atom} was not found.')

            labels[atom] = ''.join([str(x) for x in atom])

        atoms_formated.append(atom)

    return atoms_formated, coordinates, labels

# Read protein atoms
if protein_atoms:
    with open(protein_atoms) as paf:
        protein_atoms = json.load(paf)

if angles:
    with open(angles) as af:
        angles = json.load(af)

if atom_pairs:
    with open(atom_pairs) as apf:
        atom_pairs = json.load(apf)

csv_name = model+separator+ligand+'.csv'
scores_csv = docking_folder+'/.analysis/scores/'+csv_name
distance_csv = docking_folder+'/.analysis/atom_pairs/'+csv_name
angles_csv = docking_folder+'/.analysis/angles/'+csv_name

print(' '*20, end='\r')
print(f'Processing model {model} and ligand {ligand}', end='\n')

data = {}
data["Protein"] = []
data["Ligand"] = []
data["Pose"] = []
data["Score"] = []
data['RMSD'] = []
data['SASA'] = []

if protein_atoms:
    data["Closest distance"] = []
    data["Closest ligand atom"] = []
    data["Closest protein atom"] = []

if atom_pairs:
    distance_data = {}
    distance_data["Protein"] = []
    distance_data["Ligand"] = []
    distance_data["Pose"] = []

if angles:
    angle_data = {}
    angle_data["Protein"] = []
    angle_data["Ligand"] = []
    angle_data["Pose"] = []

pose_count = 0

protein_coordinates = {}
ligand_coordinates = {}
scores = {}
sasa = {}

# Get coordinates and scores for docked poses
for st in structure.StructureReader(mae_output):

    # Get protein structure
    if 'r_i_glide_gscore' not in st.property:

        protein_structure = st

        # Get protein coordinates
        for atom in st.atom:
            residue = atom.getResidue()
            chain = residue.chain
            residue_id = residue.resnum
            atom_name = atom.pdbname.replace(' ','')
            xyz = atom.xyz
            protein_coordinates[chain, residue_id, atom_name] = xyz

    # Work with the ligand poses
    else:
        # Update pose count
        pose_count += 1

        # Get protein coordinates
        ligand_coordinates[pose_count] = {}
        scores[pose_count] = st.property['r_i_glide_gscore']
        sasa[pose_count] = computeLigandSASA(st, protein_structure)

        element_count = {}
        for atom in st.atom:
            residue = atom.getResidue()
            chain = residue.chain
            residue_id = residue.resnum

            # Use PDB name
            atom_name = atom.pdbname.replace(' ', '')
            if atom_name == '':
                # Use atom name
                atom_name = atom.name

            # Assing atom names
            if atom_name == '':
                element_count.setdefault(atom.element, 0)
                element_count[atom.element] += 1
                atom.name = atom.element+str(element_count[atom.element])
                atom_name = atom.name

            xyz = atom.xyz
            ligand_coordinates[pose_count][(chain, residue_id, atom_name)] = xyz

        if pose_count == 1:
            reference_coordinates = [ligand_coordinates[pose_count][a] for a in ligand_coordinates[pose_count]]
            reference_coordinates = np.array(reference_coordinates)

# Analyse docking and store data
for pose in ligand_coordinates:

    # Store data
    data["Protein"].append(model)
    data["Ligand"].append(ligand)
    data["Pose"].append(pose)
    data["Score"].append(scores[pose])
    data["SASA"].append(sasa[pose])

    # Compute RMSD
    pose_coordinates = [ligand_coordinates[pose][a] for a in ligand_coordinates[pose]]
    pose_coordinates = np.array(pose_coordinates)
    rmsd = RMSD(reference_coordinates, pose_coordinates)
    data["RMSD"].append(rmsd)

    # Compute distances
    if atom_pairs:

        # Store data
        distance_data["Protein"].append(model)
        distance_data["Ligand"].append(ligand)
        distance_data["Pose"].append(pose)

        for i,atoms in enumerate(atom_pairs[model][ligand]):

            # Compute distance
            atoms, coordinates, labels = getAtomCoordinates(atoms, protein_coordinates, ligand_coordinates[pose])
            distance = np.linalg.norm(coordinates[atoms[0]]-coordinates[atoms[1]])
            label = '-'.join([labels[a] for a in atoms])

            # Append distance
            distance_data.setdefault(label, [])
            distance_data[label].append(distance)
            assert len(distance_data[label]) == len(distance_data['Pose'])

    if angles:

        # Store data
        angle_data["Protein"].append(model)
        angle_data["Ligand"].append(ligand)
        angle_data["Pose"].append(pose)

        for i,atoms in enumerate(angles[model][ligand]):

            # Compute distance
            atoms, coordinates, labels = getAtomCoordinates(atoms, protein_coordinates, ligand_coordinates[pose])
            v1 = coordinates[atoms[0]] - coordinates[atoms[1]]
            v2 = coordinates[atoms[2]] - coordinates[atoms[1]]
            cos_theta = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            angle = np.rad2deg(np.arccos(np.clip(cos_theta, -1, 1)))
            label = '-'.join([labels[a] for a in atoms])

            # Append angle
            angle_data.setdefault(label, [])
            angle_data[label].append(angle)
            assert len(angle_data[label]) == len(angle_data['Pose'])

    # Pending for an example
    if protein_atoms:
        atoms, coordinates, labels = getAtomCoordinates(protein_atoms[model][ligand], protein_coordinates, ligand_coordinates[pose])
        p_coordinates = np.array([coordinates[tuple(a)] for a in protein_atoms[model][ligand]])
        l_coordinates = np.array([ligand_coordinates[pose][a] for a in  ligand_coordinates[pose]])
        ligand_atom_names = [a[-1] for a in ligand_coordinates[pose]]
        protein_atom_names = [a for a in protein_atoms[model][ligand]]

        # Old implementation
        M = distance_matrix(p_coordinates, l_coordinates)
        data["Closest distance"].append(np.amin(M))
        data["Closest ligand atom"].append(ligand_atom_names[np.where(M == np.amin(M))[1][0]])
        data["Closest protein atom"].append(protein_atom_names[np.where(M == np.amin(M))[0][0]])

# Create dataframes
csv_name = model+separator+ligand+'.csv'

data = pd.DataFrame(data)
data.to_csv(docking_folder+'/.analysis/scores/'+csv_name, index=False)

if atom_pairs:
    distance_data = pd.DataFrame(distance_data)
    distance_data.to_csv(docking_folder+'/.analysis/atom_pairs/'+csv_name, index=False)

if angles:
    angle_data = pd.DataFrame(angle_data)
    angle_data.to_csv(docking_folder+'/.analysis/angles/'+csv_name, index=False)

print('\n')
print('Finished processing model')
