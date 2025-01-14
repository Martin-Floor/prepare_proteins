from pyrosetta import *
import argparse
import json
import numpy as np
import pandas as pd
from pyrosetta.rosetta.core.scoring import bb_rmsd

from multiprocessing import Pool, cpu_count

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('rosetta_folder', help='Path to json file containing the atom pairs to calculate distances.')
parser.add_argument('--atom_pairs', help='Path to json file containing the atom pairs to calculate distances.')
parser.add_argument('--models', help='Comma separated (no spaces) list of models to process')
parser.add_argument('--energy_by_residue', action='store_true', default=False, help='Get energy by residue information?')
parser.add_argument('--interacting_residues', action='store_true', default=False, help='Calculate interacting neighbour residues')
parser.add_argument('--protonation_states', action='store_true', default=False, help='Get the protonation states of a group of tritable residues.')
parser.add_argument('--cpus', default=None, help='Number of CPUs for parallelization')
parser.add_argument('--binding_energy', default=None, help='Calculate binding energy for the given chains.')
parser.add_argument('--query_residues',
                    help='Comma separated (no spaces) list of residues. Works together with --interacting_residues or --protonation_states')
parser.add_argument('--decompose_bb_hb_into_pair_energies', action='store_true', default=False,
                    help='Store backbone hydrogen bonds in the energy graph on a per-residue basis (this doubles the number of calculations, so is off by default).')
parser.add_argument('--verbose', action='store_true', default=False)

# Store variables
args=parser.parse_args()
rosetta_folder = args.rosetta_folder
atom_pairs = args.atom_pairs
energy_by_residue = args.energy_by_residue
interacting_residues = args.interacting_residues
protonation_states = args.protonation_states
binding_energy = args.binding_energy
verbose = args.verbose

cpus = args.cpus
if cpus != None:
    cpus = int(cpus)

models = args.models
if models != None:
    models = [m for m in models.split(',')]

query_residues = args.query_residues
if query_residues != None:
    query_residues = [int(r) for r in query_residues.split(',')]
decompose_bb_hb_into_pair_energies = args.decompose_bb_hb_into_pair_energies

if binding_energy != None:
    binding_energy_chains = binding_energy.split(',')
    binding_energy = True

# Initialise PyRosetta environment
init('-mute all')

# Get silent files
silent_file = {}
for model in os.listdir(rosetta_folder+'/output_models'):

    if models != None:
        if model not in models:
            continue

    for f in os.listdir(rosetta_folder+'/output_models/'+model):
        if f.endswith('.out'):
            silent_file[model] = rosetta_folder+'/output_models/'+model+'/'+f

if silent_file == {}:
    message = 'No silent files were found in rosetta calculation folder %s!' % rosetta_folder
    if models != []:
        message += '\nGiven models:\n'
        for m in models:
            message += '\t'+m+'\n'
    raise ValueError(message)


# Check if params were given
params = None
if os.path.exists(rosetta_folder+'/params'):
    params = rosetta_folder+'/params'

# Read atom_pairs dictionary
if atom_pairs != None:
    with open(atom_pairs) as apf:
        atom_pairs = json.load(apf)


def getPoseFromTag(tag, silent_file, params_dir=None):

    # Get param files
    params = []
    if params_dir != None:
        for p in os.listdir(params_dir):
            params.append(params_dir+'/'+p)

    #Read silent file data
    sfd = pyrosetta.rosetta.core.io.silent.SilentFileData(pyrosetta.rosetta.core.io.silent.SilentFileOptions())
    sfd.read_file(silent_file)

    ss = sfd.get_structure(tag)

    #Create auxiliary pose to work with
    temp_pose = Pose()

    # Read params files
    if params_dir != None:
        generate_nonstandard_residue_set(temp_pose, params)

    ss.fill_pose(temp_pose)

    #Assign tag as structure name
    temp_pose.pdb_info().name(tag)

    return temp_pose

def getTagsFromSilent(silent_file, params_dir=None):
    # Get param files
    params = []
    if params_dir != None:
        for p in os.listdir(params_dir):
            params.append(params_dir+'/'+p)

    #Read silent file data
    sfd = pyrosetta.rosetta.core.io.silent.SilentFileData(pyrosetta.rosetta.core.io.silent.SilentFileOptions())
    sfd.read_file(silent_file)

    return sorted(sfd.tags())

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
    for tag in sorted(sfd.tags()):

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

def readScoreFromSilent(score_file, indexing=False):
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
                    if '_' in line.split()[i]: # For models that have numeric numbers
                        scores[score].append(line.split()[i])
                    else:
                        scores[score].append(float(line.split()[i]))
                except:
                    scores[score].append(line.split()[i])

    scores.pop('SCORE:')
    for s in scores:
        if s == 'description':
            models = []
            poses = []
            for x in scores[s]:
                model, pose = '_'.join(x.split('_')[:-1]), x.split('_')[-1]
                models.append(model)
                poses.append(int(pose))
            continue
        scores[s] = np.array(scores[s])
    scores.pop('description')
    scores['Model'] = np.array(models)
    scores['Pose'] = np.array(poses)

    # Sort all values based on pose number
    for s in scores:
        scores[s] = [x for _,x in sorted(zip(scores['Pose'],scores[s]))]

    scores = pd.DataFrame(scores)

    if indexing:
        scores = scores.set_index(['Model', 'Pose'])

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

def calculateInterfaceScore(pose, target_chain):
    """
    Calculate interaface score for a specified jump.

    Paramters
    =========
    pose : rosetta.

    """

    sfxn = get_fa_scorefxn()

    interface_pose = pose.clone()

    rosetta.core.scoring.constraints.remove_constraints_of_type(interface_pose, 'AtomPair')

    jump_id = rosetta.core.pose.get_jump_id_from_chain(target_chain, interface_pose)
    chain_id = rosetta.core.pose.get_chain_id_from_chain(target_chain, interface_pose)
    chain_residues = rosetta.core.pose.get_chain_residues(interface_pose,chain_id)
    chain_residues_id = [x.seqpos() for x in list(chain_residues)]
    protein_residues_id = [r for r in range(1,interface_pose.total_residue()) if r not in  chain_residues_id]

    chains_coor = getCoordinates(interface_pose,chain_residues_id)
    protein_coor = getCoordinates(interface_pose, protein_residues_id)

    chain_centroid = np.average(chains_coor, axis=0)
    protein_centroid = np.average(protein_coor, axis=0)
    vector =  chain_centroid - protein_centroid
    vector = vector/np.linalg.norm(vector)
    vector = rosetta.numeric.xyzVector_double_t(vector[0], vector[1], vector[2])

    Ei = sfxn(interface_pose)

    chain_mover = rosetta.protocols.rigid.RigidBodyTransMover()
    chain_mover.trans_axis(vector)
    chain_mover.step_size(1000)
    chain_mover.rb_jump(jump_id)
    chain_mover.apply(interface_pose)

    Ef = sfxn(interface_pose)

    return Ei-Ef

def getCoordinates(pose, residues=None, bb_only=False, sc_only=False):
    """
    Get all the pose atoms coordinates. An optional list of residues can be given
    to limit coordinates to only include the atoms of these residues.

    Parameters
    ==========
    pose : pyrosetta.rosetta.core.pose.Pose
        Pose from which to get the atomic coordinates.
    residues : list
        An optional list of residues to only get their coordinates.
    bb_only : bool
        Get only backbone atom coordinates from the pose.
    sc_only : bool
        Get only sidechain atom coordinates from the pose.

    Returns
    =======
    coordinates : numpy.ndarray
        The pose's coordinates.
    """

    if bb_only and sc_only:
        raise ValueError('bb_only and sc_only cannot be given simultaneously!')

    coordinates = []
    for r in range(1, pose.total_residue()+1):
        # Check if a list of residue indexes is given.
        if residues != None:
            if r not in residues:
                continue

        # Get residue coordinates
        residue = pose.residue(r)
        bb_indexes = residue.all_bb_atoms()
        for a in range(1, residue.natoms()+1):

            # Skip non backbone atoms
            if bb_only:
                if a not in bb_indexes:
                    continue

            # Skip backbone atoms
            if sc_only:
                if a in bb_indexes:
                    continue

            # Get coordinates
            xyz = residue.xyz(a)
            xyz = np.array([xyz[0], xyz[1], xyz[2]])
            coordinates.append(xyz)

    coordinates = np.array(coordinates)

    return coordinates

def _calculateBE(arguments):

    tag, chain, silent_file, params = arguments
    pose = getPoseFromTag(tag, silent_file, params_dir=params)

    be = {}
    be['Model'], be['Pose'] = '_'.join(tag.split('_')[:-1]), tag.split('_')[-1]
    interface_score = calculateInterfaceScore(pose, chain)
    be['interface_score_'+chain] = interface_score

    return be

def _calculateDistances(arguments):
    """
    Calculate distances for a single pose
    """

    tag, atom_pairs, silent_file, params = arguments
    pose = getPoseFromTag(tag, silent_file, params_dir=params)

    # Add tags as model and pose to distances
    distances = {}
    distances['Model'], distances['Pose'] = '_'.join(tag.split('_')[:-1]), tag.split('_')[-1]

    # Get pose coordinates
    coordinates = getPoseCordinates(pose)

    # Get atom pair distances
    for pair in atom_pairs:
        label = 'distance_'
        label += '_'.join([str(x) for x in pair[0]])+'-'
        label += '_'.join([str(x) for x in pair[1]])

        # Add label to dictionary
        distances.setdefault(label, [])

        # Add atom pair distance
        if pair[0][0] not in coordinates:
            raise ValueError('Chain %s not found in pose %s' % (pair[0][0], tag))
        elif pair[0][1] not in coordinates[pair[0][0]]:
            raise ValueError('Residue %s not found in chain %s and pose %s' % (pair[0][1], pair[0][0], tag))
        elif pair[0][2] not in coordinates[pair[0][0]][pair[0][1]]:
            raise ValueError('Atom %s not found in residue %s of chain %s in pose %s' % (pair[0][2], pair[0][1], pair[0][0], tag))

        if pair[1][0] not in coordinates:
            raise ValueError('Chain %s not found in pose %s' % (pair[1][0], tag))
        elif pair[1][1] not in coordinates[pair[1][0]]:
            raise ValueError('Residue %s not found in chain %s and pose %s' % (pair[1][1], pair[1][0], tag))
        elif pair[1][2] not in coordinates[pair[1][0]][pair[1][1]]:
            raise ValueError('Atom %s not found in residue %s of chain %s in pose %s' % (pair[1][2], pair[1][1], pair[1][0], tag))

        # Add atom pair distance
        c1 = coordinates[pair[0][0]][pair[0][1]][pair[0][2]]
        c2 = coordinates[pair[1][0]][pair[1][1]][pair[1][2]]
        distances[label] = np.linalg.norm(c1-c2)

    return distances

def _calculateEnergyByResidue(arguments):
    """
    Calculate energy-by-residue data for a single pose
    """

    tag, silent_file, params = arguments
    pose = getPoseFromTag(tag, silent_file, params_dir=params)

    residue_energies = getResidueEnergies(pose,
    decompose_bb_hb_into_pair_energies=decompose_bb_hb_into_pair_energies)

    ebr = {}
    ebr['Model'] = []
    ebr['Pose'] = []
    ebr['Chain'] = []
    ebr['Residue'] = []

    # Add energy to data
    for r in residue_energies['total_score']:
        res_info = pose.pdb_info().pose2pdb(r)
        res = res_info.split()[0]
        chain = res_info.split()[1]

        ebr['Model'], ebr['Pose'] = '_'.join(tag.split('_')[:-1]), tag.split('_')[-1]
        ebr['Chain'].append(chain)
        ebr['Residue'].append(res)
        for st in residue_energies:
            ebr.setdefault(st, [])
            ebr[st].append(residue_energies[st][r])

    return ebr

def _calculateInteractingResidues(arguments):
    """
    Calculate interacting residues data for a single pose
    """

    tag, silent_file, params = arguments
    pose = getPoseFromTag(tag, silent_file, params_dir=params)

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
                neighbours_data['Model'],  = '_'.join(tag.split('_')[:-1])
                neighbours_data['Pose'] = tag.split('_')[-1]
                neighbours_data['Chain'].append(chain)
                neighbours_data['Residue'].append(res)
                neighbours_data['Neighbour chain'].append(n_chain)
                neighbours_data['Neighbour residue'].append(n_res)

                for st in dM[n]:
                    neighbours_data[st].append(dM[n][st][r])

    return neighbours_data


def calculate_rmsd_to_native(silent_file, reference_pdb, params_dir=None):
    rmsd = []
    initial = Pose()
    if params_dir:
        params = []
        for p in os.listdir(params_dir):
            params.append(params_dir+'/'+p)
        generate_nonstandard_residue_set(initial, params)
    pose_from_file(initial, reference_pdb)
    initial.pdb_info().name("native PDB")
    relaxed_structures = readPosesFromSilent(silent_file, params_dir)
    for num, pose in enumerate(relaxed_structures):
        print(f"calculating RMSD for pose {num}")
        rmsd.append(bb_rmsd(initial, pose))
    return rmsd


def _calculateProtonationStates(arguments):
    """
    Calculate protonation states data for a single pose
    """

    tag, silent_file, params = arguments
    pose = getPoseFromTag(tag, silent_file, params_dir=params)

    protonation_data = {}
    protonation_data['Model'] = []
    protonation_data['Pose'] = []
    protonation_data['Chain'] = []
    protonation_data['Residue'] = []
    protonation_data['Residue state'] = []

    # Only implemented for histidines (of course it will be easy todo it for other residues)
    for r in range(1, pose.total_residue()+1):
        residue =  pose.residue(r)
        resname = residue.name()
        res_info = pose.pdb_info().pose2pdb(r)
        res = res_info.split()[0]
        chain = res_info.split()[1]

        if resname.startswith('HIS'):

            if ':' in resname:
                resname = resname.split(':')[0]
            if resname == 'HIS_D':
                his_type = 'HID'
            if resname == 'HIS':
                his_type = 'HIE'

            protonation_data['Model'].append('_'.join(tag.split('_')[:-1]))
            protonation_data['Pose'].append(tag.split('_')[-1])
            protonation_data['Chain'].append(chain)
            protonation_data['Residue'].append(res)
            protonation_data['Residue state'].append(his_type)

    return protonation_data

# Create folder to store analysis dataframes
os.makedirs(rosetta_folder+'/.analysis', exist_ok=True)

# Create subfolder to store scores
scores_folder = rosetta_folder+'/.analysis/scores'
if not os.path.exists(scores_folder):
    os.mkdir(scores_folder)

# Create subfolder to binding energies
if binding_energy:
    binding_energy_folder = rosetta_folder+'/.analysis/binding_energy'
    if not os.path.exists(binding_energy_folder):
        os.mkdir(binding_energy_folder)

# Create subfolder to store distances
if atom_pairs != None:
    distances_folder = rosetta_folder+'/.analysis/distances'
    if not os.path.exists(distances_folder):
        os.mkdir(distances_folder)

# Create subfolder to store energy-by-residue data
if energy_by_residue:
    ebr_folder = rosetta_folder+'/.analysis/ebr'
    if not os.path.exists(ebr_folder):
        os.mkdir(ebr_folder)

# Create subfolder to store interacting neighbours data
if interacting_residues:
    neighbours_folder = rosetta_folder+'/.analysis/neighbours'
    if not os.path.exists(neighbours_folder):
        os.mkdir(neighbours_folder)

# Create subfolder to store protonation data
if protonation_states:
    protonation_folder = rosetta_folder+'/.analysis/protonation'
    if not os.path.exists(protonation_folder):
        os.mkdir(protonation_folder)

for model in silent_file:

    if verbose:
        print('Checking calculations for model %s' % model)

    # Check whether csv files exist
    file_exists = {}

    # Check score files

    score_file = scores_folder+'/'+model+'.csv'
    if not os.path.exists(score_file):
        # Create dictionary entries for scores
        scores = readScoreFromSilent(silent_file[model])
        scores.to_csv(score_file, index=False)
    else:
        if verbose:
            print('\tScore file %s was found' % score_file)

    if binding_energy:
        binding_energy_file = binding_energy_folder+'/'+model+'.csv'
        if not os.path.exists(binding_energy_file):
            be = {}
            be['Model'] = []
            be['Pose'] = []
            for chain in binding_energy_chains:
                be['interface_score_'+chain] = []

            file_exists['be'] = False
        else:
            if verbose:
                print('\tBinding energy score file %s was found' % binding_energy_file)
            file_exists['be'] = True

    if atom_pairs != None:
        # Check distance files
        distance_file = distances_folder+'/'+model+'.csv'
        if not os.path.exists(distance_file):
            # Create dictionary entries for distances
            distances = {}
            distances['Model'] = []
            distances['Pose'] = []
            file_exists['distances'] = False
        else:
            if verbose:
                print('\tDistance score file %s was found' % distance_file)
            file_exists['distances'] = True

    if energy_by_residue:
        # Check ebr files
        ebr_file = ebr_folder+'/'+model+'.csv'
        if not os.path.exists(ebr_file):
            # Create dictionary entries for distances
            ebr = {}
            ebr['Model'] = []
            ebr['Pose'] = []
            ebr['Chain'] = []
            ebr['Residue'] = []
            file_exists['ebr'] = False
        else:
            if verbose:
                print('\tEnergy-by-residue score file %s was found' % ebr_file)
            file_exists['ebr'] = True

    if interacting_residues:
        # Check neighbours files
        neighbours_file = neighbours_folder+'/'+model+'.csv'
        if not os.path.exists(neighbours_file):
            # Create scorefunction
            sfxn = get_fa_scorefxn()
            neighbours_data = {}
            neighbours_data['Model'] = []
            neighbours_data['Pose'] = []
            neighbours_data['Chain'] = []
            neighbours_data['Residue'] = []
            neighbours_data['Neighbour chain'] = []
            neighbours_data['Neighbour residue'] = []

            # Add energy entries
            for st in sfxn.get_nonzero_weighted_scoretypes():
                 neighbours_data[st.name] = []
            neighbours_data['total_score'] = []

            # Define residue to mutate to
            mutate_to = 'G'
            file_exists['neighbours'] = False
        else:
            if verbose:
                print('\tInteracting residues score file %s was found' % neighbours_file)
            file_exists['neighbours'] = True

    if protonation_states:
        # Check neighbours files
        protonation_file = protonation_folder+'/'+model+'.csv'
        if not os.path.exists(protonation_file):
            protonation_data = {}
            protonation_data['Model'] = []
            protonation_data['Pose'] = []
            protonation_data['Chain'] = []
            protonation_data['Residue'] = []
            protonation_data['Residue state'] = []
            file_exists['protonation'] = False
        else:
            if verbose:
                print('\tProtonation score file %s was found' % protonation_file)
            file_exists['protonation'] = True

    # Skip model processing if all files are found
    skip = [file_exists[x] for x in file_exists]
    if all(skip):
        if verbose:
            print('\tAll calculations were completed for model %s' % model)
        continue

    be_jobs = []
    distance_jobs = []
    ebr_jobs = []
    neighbours_jobs = []
    protonation_jobs = []

    # for pose in readPosesFromSilent(silent_file[model], params_dir=params):
    for tag in getTagsFromSilent(silent_file[model], params_dir=params):

        # Calculate binding energy
        if binding_energy and not file_exists['be']:
            for chain in binding_energy_chains:
                be_jobs.append([tag, chain, silent_file[model], params])

        # Calculate distances
        if atom_pairs != None and not file_exists['distances']:
            distance_jobs.append([tag, atom_pairs[model], silent_file[model], params])

        # Calculate energy by residue
        if energy_by_residue and not file_exists['ebr']:
            ebr_jobs.append([tag, silent_file[model], params])

        # Calculate neighbours
        if interacting_residues and not file_exists['neighbours']:
            neighbours_jobs.append([tag, silent_file[model], params])

        # Calculate protonation
        if protonation_states and not file_exists['protonation']:
            protonation_jobs.append([tag, silent_file[model], params])

    if cpus == None:
        cpus = cpu_count()
    pool = Pool(cpus)

    if len(be_jobs) > 0:
        if verbose:
            print('\tCalculating %s binding energies' % len(be_jobs))
        be_results = pool.map(_calculateBE, be_jobs)

        for ber in be_results:
            for k in ber:
                be[k].append(ber[k])

        be = pd.DataFrame(be)
        be.to_csv(binding_energy_file, index=False)

    if len(distance_jobs) > 0:
        if verbose:
            print('\tCalculating %s distances' % len(distance_jobs))
        distance_results = pool.map(_calculateDistances, distance_jobs)

        # Add distance values to distance dictionary
        for dr in distance_results:
            for k in dr:
                distances.setdefault(k,[])
                distances[k].append(dr[k])

        # Convert distances dictionary to DataFrame
        distances = pd.DataFrame(distances)
        distances.to_csv(distance_file, index=False)

    if len(ebr_jobs) > 0:
        if verbose:
            print('\tCalculating %s energy by residue scores' % len(ebr_jobs))
        ebr_results = pool.map(_calculateEnergyByResidue, ebr_jobs)

        # Add distance values to distance dictionary
        for ebr_result in ebr_results:
            for k in ebr_result:
                ebr.setdefault(k, [])
                ebr[k] += ebr_result[k]

        # Sort ebr data by residue index
        for s in ebr:
            ebr[s] = [x for _,x in sorted(zip(ebr['Residue'],ebr[s]))]

        # Convert ebr dictionary to DataFrame
        ebr = pd.DataFrame(ebr)
        ebr.to_csv(ebr_file, index=False)

    if len(neighbours_jobs) > 0:
        if verbose:
            print('\tCalculating %s interaction scores' % len(neighbours_jobs))
        neighbours_results = pool.map(_calculateInteractingResidues, neighbours_jobs)

        # Add distance values to distance dictionary
        for neighbours_result in neighbours_results:
            for k in neighbours_result:
                neighbours_data.setdefault(k,[])
                neighbours_data[k] += neighbours_result[k]

        # Sort neighbours data by residue index
        for s in neighbours_data:
            neighbours_data[s] = [x for _,x in sorted(zip(neighbours_data['Residue'],neighbours_data[s]))]

        # Convert neighbours dictionary to DataFrame
        neighbours_data = pd.DataFrame(neighbours_data)
        neighbours_data.to_csv(neighbours_file, index=False)

    if len(protonation_jobs) > 0:
        if verbose:
            print('\tCalculating the protonation of %s poses' % len(protonation_jobs))
        protonation_results = pool.map(_calculateProtonationStates, protonation_jobs)

        # Add distance values to distance dictionary
        for protonation_result in protonation_results:
            for k in protonation_result:
                protonation_data.setdefault(k,[])
                protonation_data[k] += protonation_result[k]

        # Sort protonation data by residue index
        for s in protonation_data:
            protonation_data[s] = [x for _,x in sorted(zip(protonation_data['Residue'], protonation_data[s]))]

        # Convert protonation dictionary to DataFrame
        protonation_data = pd.DataFrame(protonation_data)
        protonation_data.to_csv(protonation_file, index=False)
