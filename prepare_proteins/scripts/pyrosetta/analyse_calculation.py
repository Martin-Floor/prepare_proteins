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
parser.add_argument('--interacting_residues', action='store_true', default=False, help='Calculate interacting neighbour residues')
parser.add_argument('--protonation_states', action='store_true', default=False, help='Get the protonation states of a group of tritable residues.')
parser.add_argument('--query_residues',
                    help='Comma separated (no spaces) list of residues. Works together with --interacting_residues or --protonation_states')
parser.add_argument('--decompose_bb_hb_into_pair_energies', action='store_true', default=False,
                    help='Store backbone hydrogen bonds in the energy graph on a per-residue basis (this doubles the number of calculations, so is off by default).')

args=parser.parse_args()
rosetta_folder = args.rosetta_folder
atom_pairs = args.atom_pairs
energy_by_residue = args.energy_by_residue
interacting_residues = args.interacting_residues
protonation_states = args.protonation_states
query_residues = args.query_residues
if query_residues != None:
    query_residues = [int(r) for r in query_residues.split(',')]
decompose_bb_hb_into_pair_energies = args.decompose_bb_hb_into_pair_energies

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

def getResidueNeighbours(pose, residues, distance=5):
    """
    Get residues neighbours to a specific residue.

    Paramters
    =========
    pose : pyrosetta.rosetta.core.pose.Pose
        Input pose.
    residue : int
        Residue index.
    distance : float
        Distance at which neighbours are considered.
    """

    if isinstance(residues, int):
        residues = [residues]

    selector = rosetta.core.select.residue_selector.ResidueIndexSelector()
    for r in residues:
        selector.set_index(r)
    NeighborhoodResidueSelector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector
    ns = NeighborhoodResidueSelector(selector, distance=distance, include_focus_in_subset=False)
    sel_vec = ns.apply(pose)
    n_residues = [r for r in range(1, pose.total_residue()+1) if sel_vec[r]]

    return n_residues

def deltaEMutation(pose, mutant_position, mutant_aa, by_residue=False,
                   decompose_bb_hb_into_pair_energies=False):
    """
    Calculate the energy difference between the pose and its mutant.

    Parameters
    ==========
    pose : pyrosetta.rosetta.core.pose.Pose
        Input pose.
    mutant_position : int
        Residue index of the positions to mutate.
    mutant_aa : str
        One-letter code of the target mutations identity.
    by_residue : bool
        Return the energy differences decomposed by residue.
    decompose_bb_hb_into_pair_energies : bool
        Store backbone hydrogen bonds in the energy graph on a per-residue basis
        (this doubles the number of calculations, so is off by default).
    """

    # Create a copy of the pose
    clone_pose = Pose()
    clone_pose.assign(pose)

    # Mutate clone pose
    toolbox.mutants.mutate_residue(clone_pose, mutant_position, mutant_aa, pack_radius=0)

    # Create scorefunction
    sfxn = get_fa_scorefxn()

    # Score poses with scorefunction
    Eo = sfxn(pose)
    Em = sfxn(clone_pose)

    if by_residue:
        # Get the energy by residue of both poses
        Eo = getResidueEnergies(pose, decompose_bb_hb_into_pair_energies=decompose_bb_hb_into_pair_energies)
        Em = getResidueEnergies(clone_pose, decompose_bb_hb_into_pair_energies=decompose_bb_hb_into_pair_energies)

        De = {}
        for st in Eo:
            De[st] = {}
            for r in Eo[st]:
                De[st][r] = Em[st][r] - Eo[st][r]

    else:
        De = Em - Eo

    return De


# Check whether silent files will be read
read_silent = False
if atom_pairs != None or energy_by_residue or interacting_residues or protonation_states:
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
    ebr_data['chain'] = []
    ebr_data['residue'] = []

if interacting_residues:
    # Create scorefunction
    sfxn = get_fa_scorefxn()
    neighbours_data = {}
    neighbours_data['description'] = []
    neighbours_data['chain'] = []
    neighbours_data['residue'] = []
    neighbours_data['neighbour chain'] = []
    neighbours_data['neighbour residue'] = []

    # Add energy entries
    for st in sfxn.get_nonzero_weighted_scoretypes():
         neighbours_data[st.name] = []
    neighbours_data['total_score'] = []

    # Define residue to mutate to
    mutate_to = 'G'

if protonation_states:
    protonation_data = {}
    protonation_data['description'] = []
    protonation_data['chain'] = []
    protonation_data['residue'] = []
    protonation_data['residue state'] = []

for model in silent_file:

    # Get all scores from silent file
    scores = readScoreFromSilent(silent_file[model])

    # Add score terms as keys to the dictionary
    for score in scores:
        if score not in data:
            data[score] = []

    if read_silent:
        for pose in readPosesFromSilent(silent_file[model], params_dir=params):

            tag = pose.pdb_info().name()

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
                    label = 'distance_'
                    label += '_'.join([str(x) for x in pair[0]])+'-'
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
            else:
                data['description'].append(tag)
                # Add scores
                for score in scores:
                    data[score].append(scores[score].loc[pose.pdb_info().name()])

            if energy_by_residue:
                residue_energies = getResidueEnergies(pose, decompose_bb_hb_into_pair_energies=decompose_bb_hb_into_pair_energies)
                # Add energy to data
                for r in residue_energies['total_score']:
                    res_info = pose.pdb_info().pose2pdb(r)
                    res = res_info.split()[0]
                    chain = res_info.split()[1]

                    ebr_data['description'].append(tag)
                    ebr_data['chain'].append(chain)
                    ebr_data['residue'].append(res)
                    for st in residue_energies:
                        if st not in ebr_data:
                            ebr_data[st] = []
                        ebr_data[st].append(residue_energies[st][r])

            if interacting_residues:
                # Get neighbour residues for each residue
                all_residues = set()
                neighbours = {}
                for r in range(1, pose.total_residue()+1):
                    if query_residues != None:
                        if r not in query_residues:
                            continue
                    neighbours[r] = getResidueNeighbours(pose, r)
                    for n in neighbours[r]:
                        all_residues.add(n)

                # Calculate mutational energies
                dM = {}
                for i,r in enumerate(all_residues):
                    dM[r] = deltaEMutation(pose, r, mutate_to, by_residue=True,
                            decompose_bb_hb_into_pair_energies=decompose_bb_hb_into_pair_energies)

                # Add energy entries
                for r in range(1, pose.total_residue()+1):
                    if query_residues != None:
                        if r not in query_residues:
                            continue

                    res_info = pose.pdb_info().pose2pdb(r)
                    res = res_info.split()[0]
                    chain = res_info.split()[1]

                    for n in neighbours[r]:
                        # Store only interacting residues
                        if dM[n]['total_score'][r] != 0:
                            n_res_info = pose.pdb_info().pose2pdb(n)
                            n_res = res_info.split()[0]
                            n_chain = res_info.split()[1]
                            neighbours_data['description'].append(tag)
                            neighbours_data['chain'].append(chain)
                            neighbours_data['residue'].append(res)
                            neighbours_data['neighbour chain'].append(n_chain)
                            neighbours_data['neighbour residue'].append(n_res)

                            for st in dM[n]:
                                neighbours_data[st].append(dM[n][st][r])

            # Only implemented for histidines (of course it will be easy todo it for other residues)
            if protonation_states:
                for r in range(1, pose.total_residue()+1):
                    residue =  pose.residue(r)
                    resname = residue.name()
                    res_info = pose.pdb_info().pose2pdb(r)
                    res = res_info.split()[0]
                    chain = res_info.split()[1]

                    if resname.startswith('HIS'):
                        if resname == 'HIS_D':
                            his_type = 'HID'
                        if resname == 'HIS':
                            his_type = 'HIE'
                        protonation_data['description'].append(tag)
                        protonation_data['chain'].append(chain)
                        protonation_data['residue'].append(res)
                        protonation_data['residue state'].append(his_type)

    # If no atom pair is given just return the rosetta scores
    else:
        scores_data.append(readScoreFromSilent(silent_file[model], indexing=False))

# Add missing values in distance label entries
if atom_pairs != None and read_silent:
    for label in atom_pairs_labels:
        delta = len(data['description'])-len(data[label])
        for x in range(delta):
            data[label].append(None)

# Create dataframe from scores only
elif not read_silent:
    data = pd.concat(scores_data)

# Convert dictionary to DataFrame
data = pd.DataFrame(data)

# Save rosetta analysis data
data.to_csv('._rosetta_data.csv', index=False)

# Write energy by residue data
if energy_by_residue:
    ebr_data = pd.DataFrame(ebr_data)
    ebr_data.to_csv('._rosetta_energy_residue_data.csv', index=False)

if interacting_residues:
    neighbours_data = pd.DataFrame(neighbours_data)
    neighbours_data.to_csv('._rosetta_interacting_residues_data.csv', index=False)

if protonation_states:
    protonation_data = pd.DataFrame(protonation_data)
    protonation_data.to_csv('._rosetta_protonation_data.csv', index=False)
