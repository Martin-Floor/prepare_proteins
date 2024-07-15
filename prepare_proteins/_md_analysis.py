import os
import json
import numpy as np
import io
import shutil
import pandas as pd
import subprocess
import mdtraj as md
from pkg_resources import resource_stream, Requirement
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, fixed
from scipy.stats import linregress

class md_analysis:
    """
    Class for handling molecular dynamics (MD) simulations. Prepares trajectory data,
    calculates distances between atom pairs, and analyzes these distances.

    Attributes:
    md_folder (str): Path to the MD folder containing the output models.
    command (str): Command to use for executing GROMACS commands (default: 'gmx').
    output_group (str): Group from which to calculate distances (default: 'System').
    ligand (bool): Whether to include ligand in the PBC removal centering (default: False).
    remove_redundant_files (bool): Whether to remove original files after processing (default: False).
    overwrite (bool): Whether to overwrite existing processed files (default: False).
    """

    def __init__(self, md_folder, command='gmx', output_group='System', ligand=False, remove_redundant_files=False, overwrite=False):
        """
        Initializes the md_analysis class.

        Parameters:
        md_folder (str): Path to the MD folder containing the output models.
        command (str): Command to use for executing GROMACS commands.
        output_group (str): Group from which to calculate distances.
        ligand (bool): Whether to include ligand in the PBC removal centering.
        remove_redundant_files (bool): Whether to remove original files after processing.
        overwrite (bool): Whether to overwrite existing processed files.
        """
        self.models = [folder for folder in os.listdir(md_folder + '/output_models/') if os.path.isdir(md_folder + '/output_models/' + folder)]

        self.traj_paths = {'nvt': {}, 'npt': {}, 'md': {}}
        self.top_paths = {}
        self.distances = {}

        for model in self.models:
            self.traj_paths['nvt'][model] = {}
            self.traj_paths['npt'][model] = {}
            self.traj_paths['md'][model] = {}
            self.top_paths[model] = {}

            for replica in os.listdir(md_folder + '/output_models/' + model):
                # Define paths
                top_path = md_folder + '/output_models/' + model + '/' + replica + '/em/prot_em.gro'
                nvt_path = md_folder + '/output_models/' + model + '/' + replica + '/nvt/prot_nvt.xtc'
                npt_path = md_folder + '/output_models/' + model + '/' + replica + '/npt'
                md_path = md_folder + '/output_models/' + model + '/' + replica + '/md'

                # Print warnings for missing files
                if not os.path.exists(top_path):
                    self.top_paths[model][replica] = ''
                    print(f'WARNING: Topology for model {model} and replica {replica} could not be found. EM gro does not exist')

                if not os.path.exists(nvt_path):
                    self.traj_paths['nvt'][model][replica] = ''
                    print(f'WARNING: NVT trajectory for model {model} and replica {replica} could not be found. NVT xtc does not exist')

                if not os.path.exists(npt_path) or len(os.listdir(npt_path)) == 0:
                    self.traj_paths['npt'][model][replica] = ''
                    print(f'WARNING: NPT trajectory for model {model} and replica {replica} could not be found. Folder does not exist or is empty')

                if not os.path.exists(md_path) or len(os.listdir(md_path)) == 0:
                    self.traj_paths['md'][model][replica] = ''
                    print(f'WARNING: MD trajectory for model {model} and replica {replica} could not be found. Folder does not exist or is empty')

                # Get groups from topol index
                index_path = md_folder + '/' + 'output_models/' + model + '/' + replica + '/topol/index.ndx'
                group_dics = _readGromacsIndexFile(index_path)
                if output_group not in group_dics:
                    raise ValueError(f'The selected output group is not available in the topol/index.ndx file. The following groups are available: {",".join(group_dics.keys())}')

                # Remove PBC
                # In case of ligand PBC center is between protein ligand interface
                if ligand:
                    centering_selector = group_dics['Protein']
                else:
                    centering_selector = group_dics['Protein']

                # Process topology file
                if not os.path.exists(top_path.replace('.gro', '_noPBC.gro')) or overwrite:
                    os.system(f'echo {centering_selector} {group_dics[output_group]} | {command}  trjconv -s {top_path.replace(".gro", ".tpr")} -f {top_path} -o {top_path.replace(".gro", "_noPBC.gro")} -center -pbc res -ur compact -n {index_path}')
                    if remove_redundant_files:
                        os.remove(top_path)

                self.top_paths[model][replica] = top_path.replace('.gro', '_noPBC.gro')

                # Process NVT trajectory
                if not os.path.exists(nvt_path.replace('.xtc', '_noPBC.xtc')) or overwrite:
                    os.system(f'echo {centering_selector} {group_dics[output_group]} | {command}  trjconv -s {nvt_path.replace(".xtc", ".tpr")} -f {nvt_path} -o {nvt_path.replace(".xtc", "_noPBC.xtc")} -center -pbc res -ur compact -n {index_path}')
                    if remove_redundant_files:
                        os.remove(nvt_path)

                self.traj_paths['nvt'][model][replica] = nvt_path.replace('.xtc', '_noPBC.xtc')

                # Process NPT trajectory
                if not os.path.exists(npt_path + '/prot_npt_cat_noPBC.xtc') or overwrite:
                    remove_paths = []
                    for file in os.listdir(npt_path):
                        if file.endswith('.xtc') and 'noPBC' not in file:
                            path = npt_path + '/' + file
                            os.system(f'echo {centering_selector} {group_dics[output_group]} | {command}  trjconv -s {path.replace(".xtc", ".tpr")} -f {path} -o {path.replace(".xtc", "_noPBC.xtc")} -center -pbc res -ur compact -n {index_path}')
                            remove_paths.append(path)
                    os.system(f'{command} trjcat -f {npt_path}/*_noPBC.xtc -o {npt_path}/prot_npt_cat_noPBC.xtc -cat')
                    if remove_redundant_files:
                        [os.remove(path) for path in remove_paths]

                self.traj_paths['npt'][model][replica] = npt_path + '/prot_npt_cat_noPBC.xtc'

                # Process MD trajectory
                if not os.path.exists(md_path + '/prot_md_cat_noPBC.xtc') or overwrite:
                    remove_paths = []
                    for file in os.listdir(md_path):
                        if file.endswith('.xtc') and 'noPBC' not in file:
                            path = md_path + '/' + file
                            os.system(f'echo {centering_selector} {group_dics[output_group]} | {command}  trjconv -s {path.replace(".xtc", ".tpr")} -f {path} -o {path.replace(".xtc", "_noPBC.xtc")} -center -pbc res -ur compact -n {index_path}')
                            remove_paths.append(path)
                    os.system(f'{command} trjcat -f {md_path}/*_noPBC.xtc -o {md_path}/prot_md_cat_noPBC.xtc -cat')
                    if remove_redundant_files:
                        [os.remove(path) for path in remove_paths]

                self.traj_paths['md'][model][replica] = md_path + '/prot_md_cat_noPBC.xtc'

    def calculateDistances(self, metrics, step='md', overwrite=False):
        """
        Calculates distances between specified atom pairs over the trajectory and stores these distances.

        Parameters:
        metrics (dict): Dictionary defining atom pairs for distance calculations.
            Example:
            metrics = {
                'model1': {
                    'metric1': [((10, 'CA'), (20, 'CA'))],
                    'metric2': [((30, 'CA'), (40, 'CA'))]
                }
            }
            Each metric is a key with a list of tuples specifying residue number and atom name pairs.
        step (str): MD simulation step to use for calculations (default: 'md').
        overwrite (bool): Whether to overwrite existing distance data (default: False).
        """
        def euclidean(XA, XB, *, out=None):
            """Calculates the Euclidean distance between two points."""
            return np.sqrt(np.add.reduce(np.square(XA - XB), 1), out=out)

        for model in self.traj_paths[step]:
            if model not in self.distances:
                self.distances[model] = {}

            for replica in self.traj_paths[step][model]:
                if replica not in self.distances[model] or overwrite:
                    # Check if trajectory and topology files were generated correctly
                    if not os.path.exists(self.traj_paths[step][model][replica]):
                        print(f'WARNING: trajectory file for model {model} and replica {replica} does not exist.')
                        continue
                    if not os.path.exists(self.top_paths[model][replica]):
                        print(f'WARNING: topology file for model {model} and replica {replica} does not exist.')
                        continue

                    traj = md.load(self.traj_paths[step][model][replica], top=self.top_paths[model][replica])
                    top = traj.topology

                    distance = {}
                    for m in metrics[model]:
                        joined_distances = []
                        for d in metrics[model][m]:
                            atom1 = top.select(f'resSeq {d[0][0]} and name {d[0][1]}')
                            atom2 = top.select(f'resSeq {d[1][0]} and name {d[1][1]}')

                            atom1_xyz = traj.xyz[:, atom1[0]]
                            atom2_xyz = traj.xyz[:, atom2[0]]

                            joined_distances.append(euclidean(atom1_xyz, atom2_xyz))

                        distance[m] = [min(values) for values in zip(*joined_distances)]

                    self.distances[model][replica] = distance

    def setupCalculateDistances(self, metrics, step='md', job_folder='MD_analysis_data', overwrite=False):
        """
        Prepares and writes scripts to calculate distances for each model/replica and organizes files into a job folder.

        Parameters:
        metrics (dict): Dictionary defining atom pairs for distance calculations.
        step (str): MD simulation step to use for calculations (default: 'md').
        job_folder (str): Folder to store job scripts and data (default: 'MD_analysis_data').
        overwrite (bool): Whether to overwrite existing files (default: False).

        Returns:
        list: List of job commands to run.

        Example:
        metrics = {
            'model1': {
                'metric1': [((10, 'CA'), (20, 'CA'))],
                'metric2': [((30, 'CA'), (40, 'CA'))]
            }
        }
        This function will prepare the necessary scripts and files in the specified job_folder for distance calculations.
        """

        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        # Get control script
        script_name = "calculateDistances.py"
        control_script = resource_stream(Requirement.parse("prepare_proteins"),
                                         "prepare_proteins/scripts/md/analysis/" + script_name)
        control_script = io.TextIOWrapper(control_script)

        # Write control script to job folder
        with open(job_folder + '/._' + script_name, 'w') as sof:
            for l in control_script:
                sof.write(l)

        jobs = []
        for model in self.traj_paths[step]:
            if not os.path.exists(job_folder + '/' + model):
                os.mkdir(job_folder + '/' + model)
            for replica in self.traj_paths[step][model]:
                # Check if trajectory and topology files were generated correctly
                if not os.path.exists(self.traj_paths[step][model][replica]):
                    print(f'WARNING: trajectory file for model {model} and replica {replica} does not exist.')
                    continue
                if not os.path.exists(self.top_paths[model][replica]):
                    print(f'WARNING: topology file for model {model} and replica {replica} does not exist.')
                    continue

                job_traj_path = model + '/' + replica + '/' + self.traj_paths[step][model][replica].split('/')[-1]
                job_top_path = model + '/' + replica + '/' + self.top_paths[model][replica].split('/')[-1]

                # Copy trajectory and topology files
                if not os.path.exists(job_folder + '/' + model + '/' + replica):
                    os.mkdir(job_folder + '/' + model + '/' + replica)
                if not os.path.exists(job_folder + '/' + job_traj_path):
                    shutil.copyfile(self.traj_paths[step][model][replica], job_folder + '/' + job_traj_path)
                if not os.path.exists(job_folder + '/' + job_top_path):
                    shutil.copyfile(self.top_paths[model][replica], job_folder + '/' + job_top_path)

                new_path = job_folder + '/' + model + '/' + replica

                if not os.path.exists(new_path + '/dist.json') or overwrite:
                    command = f'cd {job_folder}/{model}/{replica}/\n'
                    command += f'python ../../._{script_name} '
                    command += f'--trajectory {self.traj_paths[step][model][replica].split("/")[-1]} '
                    command += f'--topology {self.top_paths[model][replica].split("/")[-1]} '
                    if metrics[model] is not None:
                        with open(new_path + '/metrics.json', 'w') as f:
                            json.dump(metrics[model], f)
                        command += '--metrics metrics.json '
                    command += '\n'
                    command += 'cd ..\n'
                    jobs.append(command)
        return jobs

    def readDistances(self, folder='MD_analysis_data'):
        """
        Reads pre-calculated distances from a specified folder and stores them in the class.

        Parameters:
        folder (str): Folder containing pre-calculated distance data.
        """

        if not os.path.exists(folder):
            os.mkdir(folder)

        for model in self.top_paths:
            self.distances[model] = {}
            for replica in self.top_paths[model]:
                if os.path.exists(folder + '/' + model + '/' + replica + '/dist.json'):
                    with open(folder + '/' + model + '/' + replica + '/dist.json', 'r') as f:
                        self.distances[model][replica] = json.load(f)
                    print(f'Reading distance for {model}')
                    continue

    def plot_distances(self, threshold=0.45, lim=True):
        """
        Plots distances over time for visual inspection using matplotlib.

        Parameters:
        threshold (float): Threshold to highlight in the plot (default: 0.45).
        lim (bool): Whether to limit the y-axis (default: True).
        """

        def options1(model):
            if self.distances[model]:
                for replica in self.distances[model]:
                    for metric in self.distances[model][replica]:
                        y = np.array(self.distances[model][replica][metric]) * 10
                        x = [x * 0.1 for x in range(len(y))]
                        plt.plot(x, y)

                    plt.axhline(y=threshold, color='r', linestyle='-')
                    if lim:
                        plt.ylim(1, 10)

                    plt.title(replica)
                    plt.xlabel("time (ns)")
                    plt.ylabel("distance (Å)")
                    plt.legend(list(self.distances[model][replica].keys()))
                    plt.show()
            else:
                print(f'No distance data for model {model}')

        interact(options1, model=sorted(self.distances.keys()))

    def get_distance_prob(self, threshold=0.45, group_metrics=None):
        """
        Computes the probability of distances being below a threshold and groups metrics if provided.

        Parameters:
        threshold (float): Distance threshold for calculating probabilities (default: 0.45).
        group_metrics (dict): Dictionary defining groups of metrics for combined probability calculations (default: None).

        Returns:
        pd.DataFrame: DataFrame containing probabilities for each model and replica.

        Example:
        group_metrics = {
            'group1': ['metric1', 'metric2'],
            'group2': ['metric3', 'metric4']
        }
        This function calculates the probability that the distance for each metric (and group of metrics) is below the threshold.
        """

        data = {}
        column_names = []
        for protein in self.distances:
            for replica in self.distances[protein]:
                masks = {}
                for metric in self.distances[protein][replica]:
                    if (protein, replica) not in data:
                        data[(protein, replica)] = []

                    n_frames = len(self.distances[protein][replica][metric])
                    metric_mask = [x < threshold for x in self.distances[protein][replica][metric]]

                    masks[metric] = (metric_mask)

                    # data[(protein,replica)].append(np.mean(self.distances[protein][replica][metric])
                    # data[(protein,replica)].append(np.std(self.distances[protein][replica][metric])
                    data[(protein, replica)].append(metric_mask.count(True) / n_frames)
                    if metric not in column_names:
                        column_names.append(metric)

                if group_metrics:
                    for group in group_metrics:
                        group_mask = [True] * n_frames
                        for metric in masks:
                            if metric in group_metrics[group]:
                                group_mask = [x and y for x, y in zip(masks[metric], group_mask)]

                        data[(protein, replica)].append(group_mask.count(True) / n_frames)
                        if group not in column_names:
                            column_names.append(group)

        df = pd.DataFrame(data).transpose()
        df.columns = column_names

        return df

    def plot_rmsd(self, step='md', reference=None, align=True, lim=True, order_by_protein_rmsd_slope=False, order_by_protein_rmsd_avg=False, order_by_protein_rmsd_std=False, overwrite=False):
        """
        Plots the RMSD of the protein over time for visual inspection using matplotlib.

        Parameters:
        step (str): MD simulation step to use for calculations (default: 'md').
        reference (str): Path to the reference structure file (default: None, uses the first frame of the trajectory).
        align (bool): Whether to align the trajectory to the reference structure (default: True).
        lim (bool): Whether to limit the y-axis (default: True).
        order_by_protein_rmsd_slope (bool): Whether to order the plots by the protein RMSD slope data stored in self.rmsd_trend_analysis (default: False).
        order_by_protein_rmsd_avg (bool): Whether to order the plots by the protein RMSD average data stored in self.rmsd_trend_analysis (default: False).
        order_by_protein_rmsd_std (bool): Whether to order the plots by the protein RMSD standard deviation data stored in self.rmsd_trend_analysis (default: False).
        overwrite (bool): Whether to force recalculation of RMSD trends (default: False).
        """
        def calculate_rmsd(traj, reference):
            if align:
                traj.superpose(reference)
            rmsd = md.rmsd(traj, reference)
            return rmsd

        def plot_rmsd_for_model(model):
            if self.traj_paths[step].get(model):
                for replica in self.traj_paths[step][model]:
                    traj_path = self.traj_paths[step][model][replica]
                    top_path = self.top_paths[model][replica]

                    if not traj_path or not top_path:
                        print(f'WARNING: trajectory or topology file for model {model} and replica {replica} does not exist.')
                        continue

                    traj = md.load(traj_path, top=top_path)
                    ref_structure = md.load(reference) if reference else traj[0]

                    rmsd = calculate_rmsd(traj, ref_structure)
                    x = np.arange(len(rmsd)) * traj.timestep / 1000  # Convert to nanoseconds

                    plt.plot(x, rmsd, label=replica)

                if lim:
                    plt.ylim(0, np.max(rmsd) * 1.1)

                plt.title(f'RMSD over time for model {model}')
                plt.xlabel("Time (ns)")
                plt.ylabel("RMSD (nm)")
                plt.legend()
                plt.show()
            else:
                print(f'No trajectory data for model {model}')

        # Check if rmsd_trend_analysis needs to be computed
        if (order_by_protein_rmsd_slope or order_by_protein_rmsd_avg or order_by_protein_rmsd_std) and (not hasattr(self, 'rmsd_trend_analysis') or overwrite):
            print("Computing RMSD trend analysis...")
            self.analyze_rmsd_trends(step=step, reference=reference, align=align, threshold=0.0001, overwrite=overwrite)

        # Determine the order of models
        if order_by_protein_rmsd_slope and hasattr(self, 'rmsd_trend_analysis'):
            sorted_models = self.rmsd_trend_analysis.sort_values(by='Slope', ascending=False).index.get_level_values('Model').unique()
        elif order_by_protein_rmsd_avg and hasattr(self, 'rmsd_trend_analysis'):
            sorted_models = self.rmsd_trend_analysis.sort_values(by='RMSD_avg', ascending=False).index.get_level_values('Model').unique()
        elif order_by_protein_rmsd_std and hasattr(self, 'rmsd_trend_analysis'):
            sorted_models = self.rmsd_trend_analysis.sort_values(by='RMSD_std', ascending=False).index.get_level_values('Model').unique()
        else:
            sorted_models = sorted(self.traj_paths[step].keys())

        interact(plot_rmsd_for_model, model=sorted_models)

    def analyze_rmsd_trends(self, step='md', reference=None, align=True, threshold=0.0001, sort_by_slope=True, overwrite=False):
        """
        Analyzes the RMSD progression trends and returns the results as a DataFrame.

        Parameters:
        step (str): MD simulation step to use for calculations (default: 'md').
        reference (str): Path to the reference structure file (default: None, uses the first frame of the trajectory).
        align (bool): Whether to align the trajectory to the reference structure (default: True).
        threshold (float): Threshold for classifying the RMSD trend (default: 0.0001).
        sort_by_slope (bool): Whether to sort the resulting DataFrame by slope (default: True).
        overwrite (bool): Whether to force recalculation of RMSD trends (default: False).

        Returns:
        pd.DataFrame: DataFrame containing the trend analysis results for each model and replica.
        """
        if hasattr(self, 'rmsd_trend_analysis') and not overwrite:
            print("RMSD trend analysis already exists. Use overwrite=True to force recalculation.")
            return self.rmsd_trend_analysis

        def calculate_rmsd(traj, reference):
            if align:
                traj.superpose(reference)
            rmsd = md.rmsd(traj, reference)
            return rmsd

        classifications = []
        total_tasks = sum(len(self.traj_paths[step][model]) for model in self.traj_paths[step])
        current_task = 0

        for model in self.traj_paths[step]:
            for replica in self.traj_paths[step][model]:
                current_task += 1
                message = f'Calculating RMSD for model {model}, replica {replica}... ({current_task}/{total_tasks})'
                print(message, end='\r')
                traj_path = self.traj_paths[step][model][replica]
                top_path = self.top_paths[model][replica]

                if not traj_path or not top_path:
                    print(f'WARNING: trajectory or topology file for model {model} and replica {replica} does not exist.')
                    continue

                traj = md.load(traj_path, top=top_path)
                ref_structure = md.load(reference) if reference else traj[0]

                rmsd = calculate_rmsd(traj, ref_structure)
                x = np.arange(len(rmsd)) * traj.timestep / 1000  # Convert to nanoseconds

                # Calculate RMSD statistics
                rmsd_avg = np.mean(rmsd)
                rmsd_std = np.std(rmsd)

                # Classify RMSD trend
                slope, intercept, r_value, p_value, std_err = linregress(x, rmsd)

                if slope > threshold:
                    trend = 'increasing'
                elif slope < -threshold:
                    trend = 'decreasing'
                elif std_err < threshold:
                    trend = 'stable'
                else:
                    trend = 'fluctuating'

                classifications.append({
                    'Model': model,
                    'Replica': replica,
                    'Trend': trend,
                    'Slope': slope,
                    'Intercept': intercept,
                    'R-squared': r_value**2,
                    'RMSD_avg': rmsd_avg,
                    'RMSD_std': rmsd_std
                })
                print(' ' * (len(message) + 20), end='\r')

        print()  # Print a newline at the end
        df = pd.DataFrame(classifications)
        df.set_index(['Model', 'Replica'], inplace=True)

        if sort_by_slope:
            df = df.sort_values(by='Slope', ascending=False)

        self.rmsd_trend_analysis = df
        return df

def _readGromacsIndexFile(file):
    """
    Reads a GROMACS index file and returns a dictionary of groups.

    Parameters:
    file (str): Path to the GROMACS index file.

    Returns:
    dict: Dictionary with group names as keys and their indices as values.
    """
    with open(file, 'r') as f:
        groups = [x.replace('[', '').replace(']', '').replace('\n', '').strip() for x in f.readlines() if x.startswith('[')]

    group_dict = {g: str(i) for i, g in enumerate(groups)}
    return group_dict
