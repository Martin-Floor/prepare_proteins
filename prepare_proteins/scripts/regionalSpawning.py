import os
import json

import pandas as pd
import numpy as np

import mdtraj as md
from Bio import PDB

import argparse

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('metrics', default=None, help='Path to the JSON file contanining the regional definitions')
parser.add_argument('metrics_thresholds', default=None, help='Path to the JSON file with the thresholds for defining the regions.')
parser.add_argument('--separator', default='_', help='Separator used for the protein and ligand for file names')
parser.add_argument('--max_spawnings', default=10, help='Maximum allowed spawings')
args=parser.parse_args()

### Define functions
with open(args.metrics) as jf:
    metrics = json.load(jf)
with open(args.metrics_thresholds) as jf:
    metrics_thresholds = json.load(jf)
separator = args.separator
max_spawnings = args.max_spawnings # Not used yet
verbose = True
cwd = os.getcwd()

original_yaml = cwd+'/0/input.yaml'

# Get model base name
for f in os.listdir(cwd+'/0'):
    if f.endswith('.pdb'):
        protein, ligand, pose = f.split(separator)
        pose = pose.replace('.pdb', '')

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
    for e in os.listdir(spawning_output_dir):
        try:
            spawning_epochs_paths.append(int(e))
        except:
            continue
    return {e:spawning_output_dir+'/'+str(e) for e in sorted(spawning_epochs_paths)}

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

def checkIteration(epoch_folder, metrics, metrics_thresholds, theta=0.5, fraction=0.5, verbose=True):
                   # late_arrival=0.2, conditional=0.1):
    """
    Check iteration acceptance probability for the defined regions.
    """

    # Get iteration data
    report_files = getReportFiles(epoch_folder)
    report_data = readIterationFiles(report_files)

    # Add metrics to dataframe
    for m in metrics:
        distances = [d for d in metrics[m]]
        report_data[m] = report_data[distances].min(axis=1).tolist()

    # Compute target region probability by trajectory
    probability = {}
    conditional = {}
    for t in report_data.index.levels[0]:

        # Filter data by all metric thresholds
        trajectory_data = report_data[report_data.index.get_level_values('Trajectory') == t]

        # Compute acceptance
        acceptance = np.ones(trajectory_data[m].shape, dtype=bool)
        for m in metrics:
            acceptance = acceptance & ((trajectory_data[m] <= metrics_thresholds[m]).to_numpy())

        # Compute conditional probability
        prior = np.concatenate([np.array([True]),acceptance[:-1]])
        if np.sum(~acceptance) == 0:
            conditional[t] = 0
        else:
            conditional[t] = np.sum((prior) & (~acceptance))/np.sum(~acceptance)

        # Compute trajectory probability
        probability[t] = np.sum(acceptance)/trajectory_data.shape[0]

    # Compute overall simulation probability
    P = sum([1.0 for t in probability if probability[t] >= theta])/len(probability)

    if verbose:
        print(f'The fraction of trajectories in the region is {P}')

    accepted_iteration = True
    best_pose = None
    if P < fraction:
        if verbose:
            print('Continuation was rejected')

        # Find best poses iteratively
        best_pose = np.empty(0) # Placeholder
        step = 0.1
        while best_pose.shape[0] == 0:

            # Filter dataframe by metrics' thresholds
            filtered = report_data
            metric_acceptance = {}
            for m in metrics:

                # Skip metric filters that do not have a defined threshold
                if m not in metrics_thresholds:
                    continue

                metric_acceptance[m] = report_data[report_data[m] <= metrics_thresholds[m]].shape[0]
                filtered = filtered[filtered[m] <= metrics_thresholds[m]]

            best_pose = filtered.nsmallest(1, 'Binding Energy')

            # If the pose was not found, update the metric with lowest acceptance
            if best_pose.shape[0] == 0:
                lowest_metric = [m for m,a in sorted(metric_acceptance.items(), key=lambda x:x[1])][0]
                metrics_thresholds[lowest_metric] += step

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

def getTopologyFile():
    """
    Get the tolopogy file for the PELE run
    """
    for f in os.listdir(cwd+'/0/output/input/'):
        if f.endswith('_processed.pdb'):
            return cwd+'/0/output/input/'+f

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

# Check  first iteration
spawning_indexes = getSpawningIndexes()

# Check last iteration for the last spawning
current_spawning = spawning_indexes[-1]

while current_spawning <= 10:

    # Get last epoch for the current spawning
    epochs_paths = getSpawningEpochPaths(current_spawning)
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
                restart = False
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
                if not adaptive_restart:
                    restart_yaml.write('adaptive_restart: true\n')
            restart_yaml.close()

        command = 'cd '+str(current_spawning)+'\n'
        command += 'python -m pele_platform.main input_restart.yaml\n'
        command += 'cd ..\n'
        os.system(command)

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
        with open(original_yaml) as yf:
            for l in yf:
                if l.startswith('iterations:'):
                    l = 'iterations: 1\n'
                elif l.startswith('equilibration:'):
                    l = 'equilibration: false\n'
                elif l.startswith('equilibration_steps:') or l.startswith('equilibration_mode:'):
                    continue
                elif l.startswith('debug:'):
                    continue
                new_yaml.write(l)
        new_yaml.close()

        # Run next spawning
        command = 'cd '+str(current_spawning)+'\n'
        command += 'python -m pele_platform.main input.yaml\n'
        command += 'cd ..\n'
        os.system(command)

if verbose:
    print('Maximum spawnings {max_spawnings} reached.')
    print('The regional spawning scheme has finished')
