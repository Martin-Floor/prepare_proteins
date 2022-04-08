from pyrosetta import *
import argparse
import json
import numpy as np
import pandas as pd

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('rosetta_folder', help='Path to json file containing the atom pairs to calculate distances.')
parser.add_argument('--atom_pairs', help='Path to json file containing the atom pairs to calculate distances.')
parser.add_argument('--energy_by_residue', action='store_true', default=False, help='Get energy by residue information?')

args=parser.parse_args()
rosetta_folder = args.rosetta_folder
atom_pairs = args.atom_pairs
energy_by_residue = args.energy_by_residue

# Initialise PyRosetta environment
init()

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

# Read atom_pairs dictionary
if atom_pairs != None:
    with open(atom_pairs) as apf:
        atom_pairs = json.load(apf)

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

def getPoseCordinates(pose):
    """
    Generate a dictionary to call specific coordinates from the poses. The dictionary
    is called by chain, then residue, and, finally, the atom name.

    Parameters
    ==========
    pose : pyrosetta.rosetta.core.pose.Pose
        Input pose.

    Returns
    =======
    coordinates : dict
        Pose coordinates
    """

    # Get chain letter in pose
    chain_ids = rosetta.core.pose.get_chains(pose)
    chains = {index:rosetta.core.pose.get_chain_from_chain_id(index, pose) for index in chain_ids}

    coordinates = {}

    # Iterate residues
    for chain in chains:

        # Define chain entry by coordinates
        coordinates[chains[chain]] = {}

        #Return list of residues as list
        lrb = pose.conformation().chain_begin(chain)
        lre = pose.conformation().chain_end(chain)
        residues = range(lrb, lre+1)

        for r in residues:
            resseq = pose.pdb_info().number(r)
            coordinates[chains[chain]][resseq] = {}
            for i in range(1, pose.residue_type(r).natoms()+1):
                atom_name = pose.residue_type(r).atom_name(i).strip()
                coordinates[chains[chain]][resseq][atom_name] = np.array(pose.residue(r).atom(i).xyz())

    return coordinates

def readScoreFromSilent(score_file, indexing=True):
    """
    Generates an iterator from the poses in the silent file

    Arguments:
    ==========
    silent_file : (str)
        Path to the input silent file.

    Returns:
    ========
        Generator object for the poses in the silen file.
    """
    with open(score_file) as sf:
        lines = [x for x in sf.readlines() if x.startswith('SCORE:')]
        score_terms = lines[0].split()
        scores = {}
        for line in lines[1:]:
            for i, score in enumerate(score_terms):
                if score not in scores:
                    scores[score] = []
                try:
                    scores[score].append(float(line.split()[i]))
                except:
                    scores[score].append(line.split()[i])
    scores.pop('SCORE:')
    for s in scores:
        scores[s] = np.array(scores[s])

    scores = pd.DataFrame(scores)

    if indexing:
        scores = scores.set_index('description')

    return scores

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

# Check whether silent files will be read
read_silent = False
if atom_pairs != None or energy_by_residue:
    read_silent = True

# Create dictionary entries for data
data = {}
data['description'] = []

atom_pairs_labels = []
scores_data = []
index_count = 0

if energy_by_residue:
    # Create dictionary entries
    ebr_data = {}
    ebr_data['description'] = []
    ebr_data['residue'] = []

for model in silent_file:

    # Get all scores from silent file
    scores = readScoreFromSilent(silent_file[model])

    # Add score terms as keys to the dictionary
    for score in scores:
        if score not in data:
            data[score] = []

    if read_silent:
        for i,pose in enumerate(readPosesFromSilent(silent_file[model], params)):

            if atom_pairs != None:
                # Get pose coordinates
                coordinates = getPoseCordinates(pose)

                # Update count and create index description entry
                index_count += 1
                data['description'].append(pose.pdb_info().name())

                # Add scores
                for score in scores:
                    data[score].append(scores[score].loc[pose.pdb_info().name()])

                # Get atom pair distances
                for pair in atom_pairs[model]:
                    label = '_'.join([str(x) for x in pair[0]])+'-'
                    label += '_'.join([str(x) for x in pair[1]])

                    # Add label to dictionary if not in it
                    if label not in data:
                        data[label] = []
                        atom_pairs_labels.append(label)

                    # Fill with None until match the index count
                    delta = index_count-len(data[label])
                    for x in range(delta-1):
                        data[label].append(None)

                    # Add atom pair distance
                    c1 = coordinates[pair[0][0]][pair[0][1]][pair[0][2]]
                    c2 = coordinates[pair[1][0]][pair[1][1]][pair[1][2]]
                    data[label].append(np.linalg.norm(c1-c2))

                    # Assert same length for label data
                    assert len(data[label]) == len(data['description'])

            if energy_by_residue:
                tag = pose.pdb_info().name()
                residue_energies = getResidueEnergies(pose)
                # Add energy to data
                for r in residue_energies['total_score']:
                    ebr_data['description'].append(tag)
                    ebr_data['residue'].append(r)
                    for st in residue_energies:
                        if st not in ebr_data:
                            ebr_data[st] = []
                        ebr_data[st].append(residue_energies[st][r])

            break

    # If no atom pair is given just return the rosetta scores
    else:
        print(scores_data)
        scores_data.append(readScoreFromSilent(silent_file[model], indexing=False))

# Add missing values in distance label entries
if atom_pairs != None:
    for label in atom_pairs_labels:
        delta = len(data['description'])-len(data[label])
        for x in range(delta):
            data[label].append(None)

    # Convert dictionary to DataFrame
    data = pd.DataFrame(data)
# Create dataframe from scores only
else:
    data = pd.concat(scores_data)

# Save rosetta analysis data
data.to_csv('._rosetta_data.csv', index=False)

# Write energy by residue data
if energy_by_residue:
    # Write pandas dictionary
    ebr_data = pd.DataFrame(ebr_data)
    ebr_data.to_csv('._rosetta_energy_residue_data.csv', index=False)
