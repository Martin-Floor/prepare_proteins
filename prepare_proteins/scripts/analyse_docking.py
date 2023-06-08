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
parser.add_argument('--atom_pairs', default=None, help='Dictionary json file including protein_atoms for distance calculations.')
parser.add_argument('--skip_chains', default=False, action='store_true', help='Skip chain comparison and select only by residue ID and atom name.')
parser.add_argument('--return_failed', default=False, action='store_true', help='Output a file containing failed docking models.')
parser.add_argument('--ignore_hydrogens', default=False, action='store_true', help='Ignore hydrogens for closes distance calculation.')
parser.add_argument('--separator', default='-', help='Separator to use in naming model+ligand files.')
parser.add_argument('--overwrite', default=False, action='store_true', help='Reanalyse docking simulations?')

args=parser.parse_args()

docking_folder = args.docking_folder
protein_atoms = args.protein_atoms
atom_pairs = args.atom_pairs
skip_chains = args.skip_chains
return_failed = args.return_failed
ignore_hydrogens = args.ignore_hydrogens
separator = args.separator
overwrite = args.overwrite

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
for model in os.listdir(docking_folder+'/output_models'):

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
                csv_name = model+separator+ligand+'.csv'
                if not os.path.exists(docking_folder+'/.analysis/atom_pairs/'+csv_name) or overwrite:
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
if protein_atoms != None:
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

                    distance_data["Protein"].append(model)
                    distance_data["Ligand"].append(ligand)
                    distance_data["Pose"].append(pose_count)

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

                                        # Append distance
                                        distance_data.setdefault(label, [])
                                        distance_data[label].append(d)
                                        assert len(distance_data[label]) == len(distance_data['Pose'])

                            elif isinstance(pair[1], (list,tuple)):
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
                                        distance_data.setdefault(label, [])
                                        distance_data[label].append(d)

                                        # Assert same length for label data
                                        assert len(distance_data[label]) == len(distance_data['Pose'])

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
                        p_found = []
                        for pair in atom_pairs[model][ligand]:
                            if skip_chains:
                                residue_id = pair[0][0]
                                atom_name = pair[0][1]
                                for residue in chain.residue:
                                    if residue.resnum == residue_id:
                                        for atom in residue.atom:
                                            if atom.pdbname.strip() == atom_name:
                                                p_coordinates.append(atom.xyz)
                                                p_found.append(atom)
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
                                                        p_found.append([atom.chain, atom.getResidue().resnum, atom.pdbname.strip()])

                        # Check if some protein atoms was not found
                        not_found = []
                        if len(p_found) != len(atom_pairs[model][ligand]):
                            for p in atom_pairs[model][ligand]:
                                if p[0] not in p_found:
                                    not_found.append(p[0])
                            raise ValueError('The following atoms were not found: %s' % not_found)

                        if p_coordinates == []:
                            raise ValueError('Error. Protein atoms were not found in model %s!' % model)
                        p_coordinates = np.array(p_coordinates)

            distance_data = pd.DataFrame(distance_data)
            # Create multiindex dataframe
            csv_name = model+separator+ligand+'.csv'
            distance_data.to_csv(docking_folder+'/.analysis/atom_pairs/'+csv_name, index=False)

csv_name = 'docking_data.csv'
if not os.path.exists(docking_folder+'/.analysis/'+csv_name) or overwrite:
    data = pd.DataFrame(data)
    # Create multiindex dataframe
    data.to_csv(docking_folder+'/.analysis/'+csv_name, index=False)
