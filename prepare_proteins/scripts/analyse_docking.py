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
args=parser.parse_args()

protein_atoms = args.protein_atoms
atom_pairs = args.atom_pairs
skip_chains = args.skip_chains

def RMSD(ref_coord, curr_coord):
    sq_distances = np.linalg.norm(ref_coord - curr_coord, axis=1)**2
    rmsd = np.sqrt(np.sum(sq_distances)/ref_coord.shape[0])
    return rmsd

# Read protein atoms
if protein_atoms != None:
    with open(protein_atoms) as paf:
        protein_atoms = json.load(paf)

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
for model in subjobs:
    for ligand in subjobs[model]:
        total += 1
        with open(subjobs[model][ligand]) as sjf:
            for l in sjf:
                if 'SKIP LIG' in l:
                    failed_count += 1
                    # print('Docking for %s + %s failed' % (model, ligand))

print('%s of %s models failed (%.2f%%)' % (failed_count, total, 100*failed_count/total))

data = {}
data["Protein"] = []
data["Ligand"] = []
data["Score"] = []
data["Pose"] = []
data['RMSD'] = []
if protein_atoms != None:
    data["Closest distance"] = []

# Calculate and add scores
for model in mae_output:
    if mae_output[model] != {}:
        for ligand in mae_output[model]:
            pose_count = 0
            for st in structure.StructureReader(mae_output[model][ligand]):
                if 'r_i_glide_gscore' in st.property:
                    if pose_count == 0:
                        r_coordinates = st.getXYZ()
                    pose_count += 1

                    # Store data
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

                    #Get closest ligand distance to protein atoms
                    if protein_atoms != None:
                        M = distance_matrix(p_coordinates,c_coordinates)
                        data["Closest distance"].append(np.amin(M))
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
                                        for residue in protein.residue:
                                            if residue.resnum == residue_id:
                                                for atom in residue.atom:
                                                    if atom.pdbname.strip() == atom_name:
                                                        p_coordinates.append(atom.xyz)

                        if p_coordinates == []:
                            raise ValueError('Error. Protein atoms were not found in model %s!' % model)
                        p_coordinates = np.array(p_coordinates)

data = pd.DataFrame(data)
# Create multiindex dataframe
data.to_csv('._docking_data.csv', index=False)
