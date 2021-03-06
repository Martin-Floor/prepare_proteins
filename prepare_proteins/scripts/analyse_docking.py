import os
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from schrodinger import structure
import argparse
import json

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('--protein_atoms', default=None, help='Dictionary json file including protein_atoms for distance calculations.')
parser.add_argument('--atom_pairs', default=None, help='Dictionary json file including protein_atoms for distance calculations.')
parser.add_argument('--skip_chains', default=False, action='store_true', help='Skip chain comparison and select only by residue ID and atom name.')
parser.add_argument('--return_failed', default=False, action='store_true', help='Output a file containing failed docking models.')
parser.add_argument('--ignore_hydrogens', default=False, action='store_true', help='Ignore hydrogens for closes distance calculation.')
args=parser.parse_args()

protein_atoms = args.protein_atoms
atom_pairs = args.atom_pairs
skip_chains = args.skip_chains
return_failed = args.return_failed
ignore_hydrogens = args.ignore_hydrogens

def RMSD(ref_coord, curr_coord):
    sq_distances = np.linalg.norm(ref_coord - curr_coord, axis=1)**2
    rmsd = np.sqrt(np.sum(sq_distances)/ref_coord.shape[0])
    return rmsd

# Read protein atoms
if protein_atoms != None:
    with open(protein_atoms) as paf:
        protein_atoms = json.load(paf)

if atom_pairs != None:
    with open(atom_pairs) as apf:
        atom_pairs = json.load(apf)

# Get path to outputfiles
subjobs = {}
mae_output = {}
for model in os.listdir('output_models'):
    if os.path.isdir('output_models/'+model):
        subjobs[model] = {}
        mae_output[model] = {}
        for f in os.listdir('output_models/'+model):
            if 'subjobs.log' in f:
                ligand = f.replace(model+'_','').replace('_subjobs.log','')
                subjobs[model][ligand] = 'output_models/'+model+'/'+f
            elif f.endswith('.maegz'):
                ligand = f.replace(model+'_','').replace('_pv.maegz','')
                mae_output[model][ligand] = 'output_models/'+model+'/'+f

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

print('%s of %s models failed (%.2f)' % (failed_count, total, 100*failed_count/total))

# Write failed dockings to file
if return_failed:
    with open('._failed_dockings.json', 'w') as fdjof:
        json.dump(failed_dockings, fdjof)

data = {}
data["Protein"] = []
data["Ligand"] = []
data["Score"] = []
data["Pose"] = []
data['RMSD'] = []

if protein_atoms != None:
    data["Closest distance"] = []
    data["Closest atom"] = []

if atom_pairs != None:
    atom_pairs_labels = set()

index_count = 0
# Calculate and add scores
for model in sorted(mae_output):
    if mae_output[model] != {}:
        for ligand in sorted(mae_output[model]):
            pose_count = 0
            for st in structure.StructureReader(mae_output[model][ligand]):
                if 'r_i_glide_gscore' in st.property:
                    if pose_count == 0:
                        r_coordinates = st.getXYZ()
                    pose_count += 1

                    # Store data
                    index_count += 1
                    data["Protein"].append(model)
                    data["Ligand"].append(ligand)
                    data["Pose"].append(pose_count)

                    # Get Glide score
                    score = st.property['r_i_glide_gscore']
                    data["Score"].append(score)

                    # Calcualte RMSD
                    c_coordinates = st.getXYZ()
                    rmsd = RMSD(r_coordinates, c_coordinates)
                    data["RMSD"].append(rmsd)

                    # Get atom names
                    if ignore_hydrogens:
                        selected_atoms = []
                        atom_names = []
                        for i in range(1,len(st.atom)+1):
                            if not st.atom[i].pdbname.startswith('H'):
                                selected_atoms.append(i-1)
                                atom_names.append(st.atom[i].pdbname)
                        c_coordinates = c_coordinates[selected_atoms]
                    else:
                        atom_names = []
                        for i in range(1,len(st.atom)+1):
                            atom_names.append(st.atom[i].pdbname)

                    # Calculate protein to ligand distances
                    if atom_pairs != None:
                        for i,pair in enumerate(atom_pairs[model][ligand]):
                            atom_names = [atom.pdbname.strip() for atom in st.atom]
                            if isinstance(pair[1], str):
                                ligand_atom_name = pair[1]
                                if ligand_atom_name not in atom_names:
                                    print('Ligand atoms: '+', '.join(atom_names))
                                    raise ValueError('Atom name %s not found in ligand %s' % (pair[1], ligand))
                                for j,atom in enumerate(st.atom):
                                    if atom.pdbname.strip() == pair[1]:
                                        d = np.linalg.norm(p_coordinates[i]-c_coordinates[j])
                                        label = '_'.join([str(x) for x in pair[0]])+'-'+pair[1]

                                        # Add label to dictionary if not in it
                                        if label not in data:
                                            data[label] = []
                                            atom_pairs_labels.add(label)

                                        # Fill with None until match the index count
                                        delta = index_count-len(data[label])
                                        for x in range(delta-1):
                                            data[label].append(None)

                                        # Append distance
                                        data[label].append(d)

                                        # Assert same length for label data
                                        assert len(data[label]) == len(data['Pose'])

                            elif isinstance(pair[1], tuple) or isinstance(pair[1], list):
                                ligand_chain_id = pair[1][0]
                                ligand_residue_index = pair[1][1]
                                ligand_atom_name = pair[1][2]

                                if ligand_atom_name not in atom_names:
                                    print('Ligand atoms: '+', '.join(atom_names))
                                    raise ValueError('Atom name %s not found in ligand %s' % (ligand_atom_name, ligand))
                                residue = st.findResidue(ligand_chain_id+':'+str(ligand_residue_index))
                                for j,atom in enumerate(residue.atom):
                                    if atom.pdbname.strip() == ligand_atom_name:
                                        d = np.linalg.norm(p_coordinates[i]-c_coordinates[j])
                                        label = '_'.join([str(x) for x in pair[0]])+'-'+'_'.join([str(x) for x in pair[1]])

                                        # Add label to dictionary if not in it
                                        if label not in data:
                                            data[label] = []
                                            atom_pairs_labels.add(label)

                                        # Fill with None until match the index count
                                        delta = len(data["Pose"])-len(data[label])
                                        for x in range(delta-1):
                                            data[label].append(None)
                                        data[label].append(d)

                                        # Assert same length for label data
                                        assert len(data[label]) == len(data['Pose'])

                    #Get closest ligand distance to protein atoms
                    if protein_atoms != None:
                        M = distance_matrix(p_coordinates, c_coordinates)
                        data["Closest distance"].append(np.amin(M))
                        data["Closest atom"].append(atom_names[np.where(M == np.amin(M))[1][0]])
                else:
                    if protein_atoms != None:
                        # Get protein atom coordinates
                        p_coordinates = []
                        if not isinstance(protein_atoms[model][0], list):
                            protein_atoms[model] = [protein_atoms[model]]
                        for pa in protein_atoms[model]:
                            protein = st
                            if skip_chains:
                                residue_id = pa[0]
                                atom_name = pa[1]
                            else:
                                chain_id = pa[0]
                                residue_id = pa[1]
                                atom_name = pa[2]

                            if skip_chains:
                                for residue in protein.residue:
                                    if residue.resnum == residue_id:
                                        for atom in residue.atom:
                                            if atom.pdbname.strip() == atom_name:
                                                p_coordinates.append(atom.xyz)
                            else:
                                for chain in protein.chain:
                                    if chain.name == chain_id:
                                        for residue in chain.residue:
                                            if residue.resnum == residue_id:
                                                for atom in residue.atom:
                                                    if atom.pdbname.strip() == atom_name:
                                                        p_coordinates.append(atom.xyz)

                        if p_coordinates == []:
                            raise ValueError('Error. Protein atoms were not found in model %s!' % model)
                        p_coordinates = np.array(p_coordinates)

                    elif atom_pairs != None:
                        protein = st
                        # Get protein atom coordinates
                        p_coordinates = []
                        for pair in atom_pairs[model][ligand]:
                            if skip_chains:
                                residue_id = pair[0][0]
                                atom_name = pair[0][1]
                                for residue in chain.residue:
                                    if residue.resnum == residue_id:
                                        for atom in residue.atom:
                                            if atom.pdbname.strip() == atom_name:
                                                p_coordinates.append(atom.xyz)
                            else:
                                chain_id = pair[0][0]
                                residue_id = pair[0][1]
                                atom_name = pair[0][2]
                                for chain in protein.chain:
                                    if chain.name == chain_id:
                                        for residue in chain.residue:
                                            if residue.resnum == residue_id:
                                                for atom in residue.atom:
                                                    if atom.pdbname.strip() == atom_name:
                                                        p_coordinates.append(atom.xyz)

                        if p_coordinates == []:
                            raise ValueError('Error. Protein atoms were not found in model %s!' % model)
                        p_coordinates = np.array(p_coordinates)

# Add missing values in distance label entries
if atom_pairs != None:
    for label in atom_pairs_labels:
        delta = len(data['Pose'])-len(data[label])
        for x in range(delta):
            data[label].append(None)

data = pd.DataFrame(data)
# Create multiindex dataframe
data.to_csv('._docking_data.csv', index=False)
