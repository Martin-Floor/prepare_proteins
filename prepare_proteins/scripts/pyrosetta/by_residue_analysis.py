from pyrosetta import *
import argparse
import json
import numpy as np
import pandas as pd

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('rosetta_folder', help='Path to json file containing the atom pairs to calculate distances.')
args=parser.parse_args()

# Assign variable names
rosetta_folder = args.rosetta_folder
sfxn = args.sfxn

# Get silent files
silent_file = {}
for model in os.listdir(rosetta_folder+'/output_models'):
    for f in os.listdir(rosetta_folder+'/output_models/'+model):
        if f.endswith('.out'):
            silent_file[model] = rosetta_folder+'/output_models/'+model+'/'+f

# Check if params were given
params = None
if os.path.exists(rosetta_folder+'/params'):
    params = rosetta_folder+'/params'

# Read residues dictionary
if residues != None:
    with open(residues) as rf:
        residues = json.load(rf)

def readPosesFromSilent(silent_file, params_dir=None):
    """
    Generates an iterator from the poses in the silent file

    Arguments:
    ==========
    silent_file : str
        Path to the input silent file.
    params_dir : list
        List of params files to be read if needed.

    Returns:
    ========
        Generator object for the poses in the silent file.
    """
    # Get param files
    params = []
    if params_dir != None:
        for p in os.listdir(params_dir):
            params.append(params_dir+'/'+p)

    #Read silent file data
    sfd = pyrosetta.rosetta.core.io.silent.SilentFileData(pyrosetta.rosetta.core.io.silent.SilentFileOptions())
    sfd.read_file(silent_file)

    #Iterate over tags of models, if tags is given as a list, then skip models not in tags
    for tag in sfd.tags():

        ss = sfd.get_structure(tag)

        #Create auxiliary pose to work with
        temp_pose = Pose()

        # Read params files
        if params_dir != None:
            generate_nonstandard_residue_set(temp_pose, params)

        ss.fill_pose(temp_pose)

        #Assign tag as structure name
        temp_pose.pdb_info().name(tag)

        yield temp_pose

def getResidueEnergies(pose, decompose_bb_hb_into_pair_energies=False):
    """
    Get energy by residue information for the given pose.

    Parameters
    ==========
    pose : pyrosetta.rosetta.core.pose.Pose
        Input pose to calculate energies by residues.
    decompose_bb_hb_into_pair_energies : bool
        Store backbone hydrogen bonds in the energy graph on a per-residue basis
        (this doubles the number of calculations, so is off by default).

    Returns
    =======
    residues_energies : dict
        Dictionary containing the residue energies separated by score terms.
    """
    
    sfxn = get_fa_scorefxn()

    # ### Define energy method for decomposing hbond energies into pairs
    if decompose_bb_hb_into_pair_energies:
        emo = rosetta.core.scoring.methods.EnergyMethodOptions()
        emo.hbond_options().decompose_bb_hb_into_pair_energies( True )
        sfxn.set_energy_method_options(emo)

    # Score pose
    sfxn(pose)

    # Get energy table for pose
    energies = pose.energies()

    # Get residue energies
    Er = {}
    for r in range(1, pose.total_residue()+1):
        Er[r] = energies.residue_total_energies(r)

    # Generate dictionary entries per score term and for the total score
    residues_energies = {}
    for st in sfxn.get_nonzero_weighted_scoretypes():
        residues_energies[st.name] = {}
    residues_energies['total_score'] = {}

    # Store residue energy data
    for st in sfxn.get_nonzero_weighted_scoretypes():
        for r in range(1, pose.total_residue()+1):
            residues_energies[st.name][r] = Er[r].get(st)*sfxn.get_weight(st)

            if r not in residues_energies['total_score']:
                # Add all weigthed score terms and store the residue energy
                residues_energies['total_score'][r] = np.sum([Er[r].get(st)*sfxn.get_weight(st)
                                                      for st in sfxn.get_nonzero_weighted_scoretypes()])

    return residues_energies

# Create dictionary entries
data = {}
data['description'] = []
data['residue'] = []

# Get energy by residue data
for i,pose in enumerate(readPosesFromSilent(silent_file)):
    tag = pose.pdb_info().name()
    residue_energies = getResidueEnergies(pose, sfxn)
    # Add energy to data
    for r in residue_energies['total_score']:
        data['description'].append(tag)
        data['residue'].append(r)
        for st in residue_energies:
            if st not in data:
                data[st] = []
            data[st].append(residue_energies[st][r])

# Write pandas dictionary
data = pd.DataFrame(data)
data.to_csv('._rosetta_by_residue_data.csv', index=False)
