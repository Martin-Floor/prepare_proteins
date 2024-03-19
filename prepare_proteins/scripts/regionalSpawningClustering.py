import os
import json

import pandas as pd
import numpy as np

import mdtraj as md
from Bio import PDB

import time

import argparse

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('metrics', default=None, help='Path to the JSON file contanining the regional definitions')
parser.add_argument('metrics_thresholds', default=None, help='Path to the JSON file with the thresholds for defining the regions.')
parser.add_argument('--separator', default='_', help='Separator used for the protein and ligand for file names')
parser.add_argument('--max_iterations', default=None, help='Maximum number of iterations allowed.')
parser.add_argument('--max_spawnings', default=10, help='Maximum regional spawnings allowed.')
parser.add_argument('--angles', action='store_true', default=False, help='Add angles to the PELE conf of new spawnings')

args=parser.parse_args()

### Define variables
separator = args.separator
max_iterations = args.max_iterations
max_spawnings = int(args.max_spawnings)
angles = args.angles

verbose = True
cwd = os.getcwd()

original_yaml = cwd+'/0/input.yaml'

# Get model base name
for f in os.listdir(cwd+'/0'):
    if f.endswith('.pdb'):
        protein, ligand, pose = f.split(separator)
        pose = pose.replace('.pdb', '')

# Define functions
def getSpawningIndexes():
    """
    Get the integers for the integer-named (spwaning) folders from the root directory.
    """
    spawning_folders = []
    for d in os.listdir(cwd):
        try:
            if not os.path.exists(cwd+'/'+str(d)+'/output/output'): # Do not consider spawnings without output
                continue
            spawning_folders.append(int(d))
        except:
            continue
    return sorted(spawning_folders)

def getSpawningEpochPaths(spawning_index):
    """
    Get the sorted integers for the integer-named (eochs) folders from the pele output directory.
    """
    spawning_epochs_paths = []
    spawning_output_dir = cwd+'/'+str(spawning_index)+'/output/output'

    if not os.path.exists(spawning_output_dir):
        return None

    for e in os.listdir(spawning_output_dir):
        try:
            spawning_epochs_paths.append(int(e))
        except:
            continue
    return {e:spawning_output_dir+'/'+str(e) for e in sorted(spawning_epochs_paths)}

def getTotalEpochs():
    """
    Get the total number of epochs
    """

    total_epochs = 0
    for spawning in getSpawningIndexes():
        total_epochs += len(getSpawningEpochPaths(spawning))

    return total_epochs

def getReportFiles(epoch_folder):
    """
    Gets a dictionary with all report files for the given epoch.

    Parameters
    ==========
    epoch_folder : str
        Path to the epoch folder.

    Returns
    =======
    report_files : dict
        Dictionary containing the path to all the report files (keys: trajectory_index, values: path)
    """
    rpf = []
    for f in sorted(os.listdir(epoch_folder)):
        if f.startswith('report_'):
            rpf.append((int(f.split('_')[1]), f))

    report_files = {}
    for i, f in sorted(rpf):
        report_files[i] = epoch_folder+'/'+f

    return report_files

def getTrajectoryFiles(epoch_folder):
    """
    Gets a dictionary with all report files for the given epoch.

    Parameters
    ==========
    epoch_folder : str
        Path to the epoch folder.

    Returns
    =======
    report_files : dict
        Dictionary containing the path to all the report files (keys: trajectory_index, values: path)
    """
    trf = []
    for f in sorted(os.listdir(epoch_folder)):
        if f.startswith('trajectory_') and f.endswith('.xtc'):
            trf.append((int(f.split('_')[1]), f))

    trajectory_files = {}
    for i, f in sorted(trf):
        trajectory_files[i] = epoch_folder+'/'+f

    return trajectory_files

def readIterationFiles(report_files):
    """
    Read all report files into a dataframe

    Parameters
    ==========
    report_files : dict
        Report files paths as a dictionary (see getReportFiles())

    Returns
    =======
    report_data : pandas.DataFrame
        Dataframe containing the iteration data
    """

    report_data = {}
    report_data['Trajectory'] = []
    for i in report_files:
        with open(report_files[i]) as itf:
            for l in itf:
                if l.startswith('#Task'):
                    l = l.replace('Binding Energy', 'Binding_Energy')
                    l = l.replace('BindingEnergy', 'Binding_Energy')
                    terms = l.split()
                    continue
                for t,v in zip(terms, l.split()):
                    if t in ['#Task']:
                        report_data['Trajectory'].append(i)
                        continue
                    if t == 'Binding_Energy':
                        t = 'Binding Energy'
                    if t == 'numberOfAcceptedPeleSteps':
                        t = 'Accepted PELE Step'
                    report_data.setdefault(t, [])
                    try: report_data[t].append(int(v))
                    except: report_data[t].append(float(v))


    report_data = pd.DataFrame(report_data).set_index(['Trajectory', 'Accepted PELE Step'])

    return report_data

def clusterTrajectories(trajectory_dict, report_data, metric, check_membership=None, verbose=True):
    """
    Cluster trajectories by ligand RMSD and taking as cluster centers the structure
    with the lowest value in the given metric.
    """

    Ti = time.time()

    regional_mask = report_data['Regional Acceptance'].to_numpy()

    if np.sum(regional_mask) == 0:
        print('No poses were found in region. Skipping clustering.')
        return

    if verbose:
        print(f'Clustering {np.sum(regional_mask)} poses')

    # Define topology file
    topology_file = getTopologyFile()
    topology = md.load(topology_file)

    # Load frames within region
    traj_files = [trajectory_dict[t] for t in trajectory_dict]
    traj = md.load(traj_files, top=topology_file)
    traj = traj[regional_mask]

    # Get binding energies
    energies = report_data['Binding Energy'].to_numpy()
    energies = energies[regional_mask]

    # Superpose trajectory into protein atoms
    protein_atoms = traj.topology.select('protein')
    traj.superpose(topology, atom_indices=protein_atoms)

    # Get ligand indexes
    ligand_name = getLigandResidueName()
    ligand_indexes = []
    for atom in topology.topology.atoms:
        if atom.residue.name == ligand_name:
            ligand_indexes.append(atom.index)

    # Cluster trajectories
    assigned_clusters = {}
    S = []
    for threshold in np.arange(1.0, 20.1, 0.5):
        assigned_clusters[threshold], silhouetteS = clusterByRMSD(traj,  threshold, ligand_indexes, energies)
        S.append(silhouetteS)
        n_clusters = np.max(list(assigned_clusters[threshold].values()))
        if n_clusters == 1:
            break

    print(assigned_clusters.keys())
    selected_threshold = list(assigned_clusters.keys())[np.argmax(S)]

    # Get full cluster memberships
    index = 0
    cluster_membership = []
    for i in regional_mask:
        if i:
            cluster_membership.append(assigned_clusters[selected_threshold][index])
            index += 1
        else:
            cluster_membership.append(None)

    # Check memberships
    if check_membership:
        print(check_membership)

    if verbose:
        T = time.time() - Ti
        print(f'Clustered ligand trajectories at {selected_threshold} Å RMSD in %.2f seconds.' % T)

    return cluster_membership

def clusterByRMSD(traj, rmsd_threshold, atom_indices, energies):
    """
    Cluster trajectories by RMSD. The silhouette score is computed and returned for
    later estimation of the optimal number of clusters.
    """
    # Cluster trajectories by RMSD
    assigned_indexes = {}
    cluster = 1
    cluster_center = {}
    while len(assigned_indexes) < traj.n_frames:

        index_mapping = {}
        current_indexes = []
        ni = 0
        for i in range(traj.n_frames):
            if i not in assigned_indexes:
                current_indexes.append(i)
                index_mapping[ni] = i
                ni += 1

        # Get not assigned trajectory
        rmsd_traj = traj[current_indexes]
        rmsd_energies = energies[current_indexes]

        # Get best model based on best energy
        best_model = rmsd_traj[rmsd_energies.argmin()]

        # Compute RMSD
        rmsd = md.rmsd(rmsd_traj, best_model, atom_indices=atom_indices)*10.0
        cluster_indexes = np.argwhere(rmsd <= rmsd_threshold).flatten()

        # Store index of centroid
        cluster_center[cluster] = index_mapping[rmsd_energies.argmin()]

        for i in cluster_indexes:
            assigned_indexes[index_mapping[i]] = cluster

        cluster += 1

    # Compute silhouette score

    # Compute inter cluster distances
    centers = [cluster_center[cluster] for cluster in cluster_center]
    traj_centers = traj[centers]
    B = np.zeros((traj_centers.n_frames, traj_centers.n_frames))
    for i in range(traj_centers.n_frames):
        B[i] = md.rmsd(traj_centers, traj_centers[i], atom_indices=atom_indices)*10.0
    B = np.average(B)

    # Compute inner cluster RMSD

    ### All to all implementation (consider only distance to centers for fastest computation.)
    A = np.array([])
    for cluster in sorted(set(assigned_indexes.values())):
        cluster_indexes = [i for i in assigned_indexes if assigned_indexes[i] == cluster]
        traj_clusters = traj[cluster_indexes]
        a = np.zeros((traj_clusters.n_frames, traj_clusters.n_frames))
        for i in range(traj_clusters.n_frames):
            a[i] = md.rmsd(traj_clusters, traj_clusters[i], atom_indices=atom_indices)*10.0
        A = np.concatenate([A, a.flatten()])

    A = np.average(A)
    S = (B-A)/np.max([A,B])

    return assigned_indexes, S

def checkIteration(epoch_folder, metrics, metrics_thresholds, theta=0.5, fraction=0.5, verbose=True):
                   # late_arrival=0.2, conditional=0.1):
    """
    Check iteration acceptance probability for the defined regions.
    """

    # Get iteration data
    report_files = getReportFiles(epoch_folder)
    trajectory_files = getTrajectoryFiles(epoch_folder)
    report_data = readIterationFiles(report_files)

    # Add metrics to dataframe
    metric_type = {} # Store metric type
    for m in metrics:

        # Check how metrics will be combined
        distances = False
        angles = False
        for x in metrics[m]:
            if 'distance_' in x:
                distances = True
            elif 'angle_' in x:
                angles = True

        if distances and angles:
            raise ValueError(f'Metric {m} combines distances and angles which is not supported.')

        elif distances:
            metric_type[m] = 'distance'
            # Combine distances into metrics
            distances = [d for d in metrics[m] if d.startswith('distance_')]
            report_data[m] = report_data[distances].min(axis=1).tolist()

        elif angles:
            metric_type[m] = 'angle'
            # Combine angles into metric
            angles = [d for d in metrics[m] if d.startswith('angle_')]
            if len(angles) > 1:
                raise ValueError('Combining more than one angle into a metric is not currently supported.')
            report_data[m] = report_data[angles].min(axis=1).tolist()

    # Add region membership information to report dataframe
    region_acceptance = np.ones(report_data.shape[0], dtype=bool)
    for m in metrics:

        # Skip metric filters that do not have a defined threshold
        if m not in metrics_thresholds:
            continue

        acceptance = np.ones(report_data[m].shape[0], dtype=bool)
        if isinstance(metrics_thresholds[m], float):
            acceptance = acceptance & ((report_data[m] <= metrics_thresholds[m]).to_numpy())
        elif isinstance(metrics_thresholds[m], list):
            acceptance = acceptance & ((report_data[m] >= metrics_thresholds[m][0]).to_numpy())
            acceptance = acceptance & ((report_data[m] <= metrics_thresholds[m][1]).to_numpy())
        report_data[m+' Acceptance'] = acceptance
        region_acceptance = region_acceptance & acceptance
    report_data['Regional Acceptance'] = region_acceptance

    # Compute target region probability by trajectory
    probability = {}
    conditional = {}
    for t in report_data.index.levels[0]:

        # Get trajectory data
        trajectory_data = report_data[report_data.index.get_level_values('Trajectory') == t]

        # Compute trajectory probability
        region_acceptance = trajectory_data['Regional Acceptance'] == True
        probability[t] = np.sum(region_acceptance)/trajectory_data.shape[0]

    # Compute overall simulation probability
    P = sum([1.0 for t in probability if probability[t] >= theta])/len(probability)

    if verbose:
        print(f'The fraction of trajectories in the region is %.4f' % P)

    accepted_iteration = True
    best_pose = None
    if P < fraction:
        if verbose:
            print('Continuation was rejected')

        clusters = None
        if current_spawning > 0:
            # Cluster trajectories by ligand
            if verbose:
                print('Clustering trajectories by ligand conformation')
            clusters = clusterTrajectories(trajectory_files, report_data, 'Binding Energy')

        # Find best poses iteratively
        best_pose = np.empty(0) # Placeholder
        distance_step = 0.1
        angular_step = 1.0
        while best_pose.shape[0] == 0:

            # Filter dataframe by metrics' thresholds
            filtered = report_data
            metric_acceptance = {}
            for m in metrics:

                # Skip metric filters that do not have a defined threshold
                if m not in metrics_thresholds:
                    continue

                # Filter by values lower than the given value
                if isinstance(metrics_thresholds[m], float):
                    metric_acceptance[m] = report_data[report_data[m] <= metrics_thresholds[m]].shape[0]
                    filtered = filtered[filtered[m] <= metrics_thresholds[m]]

                # Filter by values inside the two values
                elif isinstance(metrics_thresholds[m], list):
                    metric_filter = report_data[metrics_thresholds[m][0] <= report_data[m]]
                    metric_acceptance[m] = metric_filter[metric_filter[m] <= metrics_thresholds[m][1]].shape[0]
                    filtered = filtered[metrics_thresholds[m][0] <= filtered[m]]
                    filtered = filtered[filtered[m] <= metrics_thresholds[m][1]]

            best_pose = filtered.nsmallest(1, 'Binding Energy')

            # If the pose was not found, update the metric with lowest acceptance
            if best_pose.shape[0] == 0:

                lowest_metric = [m for m,a in sorted(metric_acceptance.items(), key=lambda x:x[1])][0]

                if metric_type[lowest_metric] == 'distance':
                    step = distance_step
                elif metric_type[lowest_metric] == 'angle':
                    step = angular_step

                if isinstance(metrics_thresholds[lowest_metric], float):
                    metrics_thresholds[lowest_metric] += step
                if isinstance(metrics_thresholds[lowest_metric], list):
                    metrics_thresholds[lowest_metric][0] -= step
                    metrics_thresholds[lowest_metric][1] += step

        accepted_iteration = False

    else:
        if verbose:
            print('Continuation was accepted.')

    return accepted_iteration, best_pose

def getTrajectoryFiles(iteration_folder):
    """
    Gets a dictionary with all trajectories for an iteration.

    Parameters
    ==========
    iteration_folder : str
        Path to the iteration folder.

    Returns
    =======
    trajectory_files : dict
        Dictionary containing the path to all the trajectory files (keys: trajectory index, values: path)
    """
    tjf = []
    for f in sorted(os.listdir(iteration_folder)):
        if f.endswith('.xtc'):
            tjf.append((int(f.split('_')[1].replace('.xtc', '')), f))

    trajectory_files = {}
    for i, f in sorted(tjf):
        trajectory_files[i] = iteration_folder+'/'+f

    return trajectory_files

def getTopologyFile(spawning=None):
    """
    Get the tolopogy file for the PELE run
    """
    if not spawning:
        spawning = '0'
    else:
        spawning = str(spawning)

    for f in os.listdir(cwd+'/'+spawning+'/output/input/'):
        if f.endswith('_processed.pdb'):
            return cwd+'/'+spawning+'/output/input/'+f

def getLigandFile():
    for f in os.listdir(cwd+'/0/output/input/'):
        if f == 'ligand.pdb':
            return cwd+'/0/output/input/'+f

def getLigandResidueName():
    ligand_file = getLigandFile()
    ligand_traj = md.load(ligand_file)
    ligand_residue = [r for r in ligand_traj.topology.residues][0]
    return ligand_residue.name

def extractPoses(epoch_folder, data, output_file):
    """
    Extract poses in the given dataframe
    """

    trajectory_files = getTrajectoryFiles(epoch_folder)
    topology_file = getTopologyFile()

    # Read topology as Bio.PDB.Structure
    parser = PDB.PDBParser()
    structure = parser.get_structure('topology', topology_file)

    # Give traj coordinates to PDB structure
    for t, s in data.index:
        # output_name = output_file # Redefine when implemented
        traj = md.load(trajectory_files[t], top=topology_file)
        for pdb_atom, xtc_atom in zip(structure.get_atoms(), traj.topology.atoms):
            pdb_atom.coord = traj.xyz[s][xtc_atom.index]*10.0

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)

def extendOneEpoch(spawning_index, current_epoch):
    """
    Modify adaptive.conf to extend the simulation one further epoch.
    """

    adaptive_file = cwd+'/'+str(spawning_index)+'/output/adaptive.conf'

    # Load modified pele.conf as json
    with open(adaptive_file) as ac:
        adaptive_conf = json.load(ac)

    # Update the number of iterations
    adaptive_conf['simulation']['params']['iterations'] = current_epoch+2

    # Write adaptive.conf
    with open(adaptive_file, 'w') as ac:
        json.dump(adaptive_conf, ac, indent=4)

    return current_epoch+1

### Start main here ###

# Get spawning folders indexes
spawning_indexes = getSpawningIndexes()

if not spawning_indexes:
    raise ValueError("There is not PELE output folders for any spawning folder")

# Check last spawning
current_spawning = spawning_indexes[-1]

while current_spawning <= max_spawnings:

    # Restart reading of metrics
    with open(args.metrics) as jf:
        metrics = json.load(jf)

    with open(args.metrics_thresholds) as jf:
        metrics_thresholds = json.load(jf)

    # Check if max number of iterations has been reached
    if max_iterations:
        total_iterations = getTotalEpochs()
        if total_iterations >= int(max_iterations):
            break

    # Get last epoch for the current spawning
    epochs_paths = getSpawningEpochPaths(current_spawning)

    if not epochs_paths:
        raise ValueError(f'There are not epoch folders for the current spawning {current_spawning}')

    current_epoch = list(epochs_paths.keys())[-1]

    print(f'Checking current spawning {current_spawning} and epoch {current_epoch}')

    # Check the last epoch for regional spawning logic
    accepted, best_pose = checkIteration(epochs_paths[current_epoch], metrics, metrics_thresholds)

    if accepted:

        # Extend spawning one further epoch
        new_epoch = extendOneEpoch(current_spawning, current_epoch)

        if verbose:
            print(f'Continuing with epoch {new_epoch}')

        # Check if adaptive have an adaptive restart flag
        restart_yaml = False
        if os.path.exists(cwd+'/'+str(current_spawning)+'/input_restart.yaml'):
            restart_yaml = True

        if not restart_yaml:
            restart_yaml = open(cwd+'/'+str(current_spawning)+'/input_restart.yaml', 'w')

            restart_line = False
            restart_adaptive_line = False
            with open(original_yaml) as yf:
                adaptive_restart = False
                for l in yf:
                    if l.startswith('debug:'):
                        continue
                if l.startswith('restart: true'):
                    restart_line = True
                elif l.startswith('adaptive_restart: true'):
                    restart_adaptive_line = True
                restart_yaml.write(l)
                if not restart_line:
                    restart_yaml.write('restart: true\n')
                if not restart_adaptive_line:
                    restart_yaml.write('adaptive_restart: true\n')
            restart_yaml.close()

        command = 'cd '+str(current_spawning)+'\n'
        command += 'python -m pele_platform.main input_restart.yaml\n'
        command += 'cd ..\n'
        os.system(command)

        if not os.path.exists(cwd+'/'+str(current_spawning)+'/output/output/'+str(new_epoch)):
            print(f'Something went wrong. The output for epoch {new_epoch} in spawning {current_spawning} was not found.')
            exit()

    else:
        current_spawning += 1

        if verbose:
            print(f'Configuring spawning {current_spawning}')

        if not os.path.exists(str(current_spawning)):
            os.mkdir(str(current_spawning))

        # Extract best pose to current spawning folder
        output_pdb = str(current_spawning)+'/'+protein+separator+ligand+separator+pose+'.pdb'
        extractPoses(epochs_paths[current_epoch], best_pose, output_pdb)

        # Set PELE input files
        new_yaml = open(str(current_spawning)+'/input.yaml', 'w')

        if angles:
            restart_yaml = open(cwd+'/'+str(current_spawning)+'/input_restart.yaml', 'w')
            restart_line = False
            restart_adaptive_line = False

        with open(original_yaml) as yf:
            for l in yf:
                if l.startswith('iterations:'):
                    l = 'iterations: 1\n'
                elif l.startswith('equilibration:'):
                    l = 'equilibration: false\n'
                elif l.startswith('equilibration_steps:') or l.startswith('equilibration_mode:'):
                    continue
                if angles:
                    if l.startswith('debug:'):
                        restart_yaml.write(l)
                    if l.startswith('restart: true'):
                        restart_line = True
                    elif l.startswith('adaptive_restart: true'):
                        restart_adaptive_line = True
                else:
                    if l.startswith('debug:'):
                        continue
                new_yaml.write(l)
                restart_yaml.write(l)

            if angles and not restart_line:
                restart_yaml.write('restart: true\n')
            if angles and not restart_adaptive_line:
                restart_yaml.write('adaptive_restart: true\n')

        new_yaml.close()
        if angles:
            restart_yaml.close()

        # Run next spawning
        command = 'cd '+str(current_spawning)+'\n'
        command += 'python -m pele_platform.main input.yaml\n'
        if angles:

            # Get topology
            command += 'python ../../._addAnglesToPELEConf.py output '
            command += '../0/._angles.json '
            command += '../0/output/input/'+protein+separator+ligand+separator+pose+'_processed.pdb\n'
            command += 'python -m pele_platform.main input_restart.yaml\n'
        command += 'cd ..\n'
        os.system(command)

if verbose:
    if current_spawning >= max_spawnings:
        print(f'Maximum spawnings {max_spawnings} reached.')
        print('The regional spawning scheme has finished')
    elif total_iterations >= int(max_iterations):
        print(f'Maximum iterations {max_iterations} reached.')
        print('The regional spawning scheme has finished')
