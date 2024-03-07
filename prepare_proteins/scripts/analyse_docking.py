import os
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from schrodinger import structure
import argparse
import json

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('docking_folder', default=None, help='Path to the docking folder')
parser.add_argument('--protein_atoms', default=None, help='Dictionary json file including protein_atoms for closest distance calculations.')
parser.add_argument('--atom_pairs', default=None, help='Dictionary json file including atom pairs for distance calculations.')
parser.add_argument('--angles', default=None, help='Dictionary json file including angles definitions.')
parser.add_argument('--skip_chains', default=False, action='store_true', help='Skip chain comparison and select only by residue ID and atom name.')
parser.add_argument('--return_failed', default=False, action='store_true', help='Output a file containing failed docking models.')
parser.add_argument('--ignore_hydrogens', default=False, action='store_true', help='Ignore hydrogens for closes distance calculation.')
parser.add_argument('--separator', default='-', help='Separator to use in naming model+ligand files.')
parser.add_argument('--only_models', help='Comma separated list of models to be analysed.')
parser.add_argument('--overwrite', default=False, action='store_true', help='Reanalyse docking simulations?')
args=parser.parse_args()

docking_folder = args.docking_folder
protein_atoms = args.protein_atoms
atom_pairs = args.atom_pairs
angles = args.angles
skip_chains = args.skip_chains
return_failed = args.return_failed
ignore_hydrogens = args.ignore_hydrogens
separator = args.separator
only_models = args.only_models
overwrite = args.overwrite

def RMSD(ref_coord, curr_coord):
    sq_distances = np.linalg.norm(ref_coord - curr_coord, axis=1)**2
    rmsd = np.sqrt(np.sum(sq_distances)/ref_coord.shape[0])
    return rmsd

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

# Get path to outputfiles
subjobs = {}
mae_output = {}
for model in os.listdir(docking_folder+'/output_models'):

    # Skip models not given in only_models
    if only_models != None  and model not in only_models:
        continue

    # Check separator in model name
    if separator in model:
        raise ValueError('The separator %s was found in model name %s. Please use a different one!' % (separator, model))

    if os.path.isdir(docking_folder+'/output_models/'+model):

        subjobs[model] = {}
        mae_output[model] = {}
        for f in os.listdir(docking_folder+'/output_models/'+model):

            if 'subjobs.log' in f:
                ligand = f.replace(model+'_','').replace('_subjobs.log','')
                subjobs[model][ligand] = docking_folder+'/output_models/'+model+'/'+f

            elif f.endswith('.maegz'):
                ligand = f.replace(model+'_','').replace('_pv.maegz','')

                # Check separator in ligand name
                if separator in ligand:
                    raise ValueError('The separator %s was found in ligand name %s. Please use a different one!' % (separator, ligand))

                # Check that the CSV distance files exists
                distance_csv_name = model+separator+ligand+'.csv'
                if not os.path.exists(docking_folder+'/.analysis/atom_pairs/'+distance_csv_name) or overwrite:
                    mae_output[model][ligand] = docking_folder+'/output_models/'+model+'/'+f

# Get failed models
failed_count = 0
total = 0
failed_dockings = []

failed_lines = ['SKIP LIG', 'No poses met the energy criterion for evaluating GlideScore']
for model in subjobs:
    for ligand in subjobs[model]:
        total += 1
        with open(subjobs[model][ligand]) as sjf:
            for l in sjf:
                for fl in failed_lines:
                    if fl in l:
                        failed_count += 1
                        failed_dockings.append((model, ligand))
                        print('Docking for %s + %s failed' % (model, ligand))

if total == 0:
    print('There is no docking outputs in the  given folder!')
    exit()
else:
    percentage = 100*failed_count/total
    print('%s of %s models failed (%.2f)' % (failed_count, total, percentage))

# Write failed dockings to file
if return_failed:
    with open(docking_folder+'/.analysis/._failed_dockings.json', 'w') as fdjof:
        json.dump(failed_dockings, fdjof)

data = {}
data["Protein"] = []
data["Ligand"] = []
data["Pose"] = []
data["Score"] = []
data['RMSD'] = []
if protein_atoms:
    data["Closest distance"] = []
    data["Closest atom"] = []

index_count = 0

# Calculate and add scores
for model in sorted(mae_output):

    if mae_output[model] != {}:

        for ligand in sorted(mae_output[model]):

            distance_data = {}
            distance_data["Protein"] = []
            distance_data["Ligand"] = []
            distance_data["Pose"] = []

            angle_data = {}
            angle_data["Protein"] = []
            angle_data["Ligand"] = []
            angle_data["Pose"] = []

            pose_count = 0

            protein_coordinates = {}
            ligand_coordinates = {}
            scores = {}

            # Get coordinates and scores for docked poses
            for st in structure.StructureReader(mae_output[model][ligand]):

                # Get protein structure
                if 'r_i_glide_gscore' not in st.property:

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

                    for atom in st.atom:
                        residue = atom.getResidue()
                        chain = residue.chain
                        residue_id = residue.resnum
                        atom_name = atom.pdbname.replace(' ','')
                        xyz = atom.xyz
                        ligand_coordinates[pose_count][chain, residue_id, atom_name] = xyz

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
                # if protein_atoms:
                #     for i,atoms in enumerate(protein_atoms[model][ligand]):
                #         atoms, coordinates, labels = getAtomCoordinates(atoms, protein_coordinates, ligand_coordinates[pose])
                #         pa_coordinates = np.array([coordinates[a] for a in atoms])
                #         l_coordinates = # missing an example to continue
                #
                        # Old implementation
                #         M = distance_matrix(p_coordinates, c_coordinates)
                #         data["Closest distance"].append(np.amin(M))
                #         data["Closest atom"].append(atom_names[np.where(M == np.amin(M))[1][0]])

    # Create dataframes
    csv_name = model+separator+ligand+'.csv'
    if atom_pairs:
        distance_data = pd.DataFrame(distance_data)
        distance_data.to_csv(docking_folder+'/.analysis/atom_pairs/'+csv_name, index=False)
    if angles:
        angle_data = pd.DataFrame(angle_data)
        angle_data.to_csv(docking_folder+'/.analysis/angles/'+csv_name, index=False)

csv_name = 'docking_data.csv'
if not os.path.exists(docking_folder+'/.analysis/'+csv_name) or overwrite:
    data = pd.DataFrame(data)
    data.to_csv(docking_folder+'/.analysis/'+csv_name, index=False)
