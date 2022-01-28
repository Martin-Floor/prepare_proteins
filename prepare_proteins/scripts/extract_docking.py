import os
import numpy as np
import pandas as pd
import argparse
from schrodinger import structure

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('docking_data', default=None, help='CSV file containing docking data.')
parser.add_argument('docking_folder', default=None, help='Docking output folder')

args=parser.parse_args()
docking_data = args.docking_data
docking_folder = args.docking_folder

# Read dockign data to pandas
docking_data = pd.read_csv(docking_data)
docking_data.set_index(['Protein', 'Ligand', 'Pose'], inplace=True)

# Get models list
poses = {}
for i in docking_data.index:
    model, ligand, pose = i
    if model not in poses:
        poses[model] = {}
    if ligand not in poses[model]:
        poses[model][ligand] = []
    poses[model][ligand].append(pose)

# Get path to output files
subjobs = {}
mae_output = {}
for model in os.listdir(docking_folder+'/output_models'):
    if os.path.isdir(docking_folder+'/output_models/'+model):
        subjobs[model] = {}
        mae_output[model] = {}
        for f in os.listdir(docking_folder+'/output_models/'+model):
            if 'subjobs.log' in f:
                ligand = f.replace(model+'_','').replace('_subjobs.log','')
                subjobs[model][ligand] = docking_folder+'/output_models/'+model+'/'+f
            elif f.endswith('.maegz'):
                ligand = f.replace(model+'_','').replace('_pv.maegz','')
                mae_output[model][ligand] = docking_folder+'/output_models/'+model+'/'+f

# Get and save selected docking poses to PDB
for model in poses:
    if not os.path.exists(model):
        os.mkdir(model)
    for ligand in poses[model]:
        for pose,st in enumerate(structure.StructureReader(mae_output[model][ligand])):
            if 'r_i_glide_gscore' in st.property:
                if pose in poses[model][ligand]:
                    complex = protein.merge(st)
                    output_pdb = model+'-'+ligand+'-'+str(pose)+'.pdb'
                    PDBWriter = structure.PDBWriter(model+'/'+output_pdb)
                    PDBWriter.write(complex)
            else:
                protein = st
