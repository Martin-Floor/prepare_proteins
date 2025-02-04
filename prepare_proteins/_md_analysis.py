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

    def __init__(self, md_folder, command='gmx', output_group='System', timestep=2, ligand=False, remove_redundant_files=False, overwrite=False):
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

        self.timestep = timestep

        self.traj_paths = {'nvt': {}, 'npt': {}, 'md': {}}
        self.top_paths = {}
        self.distances = {}
        self.angles = {}

        for model in self.models:
            self.traj_paths['nvt'][model] = {}
            self.traj_paths['npt'][model] = {}
            self.traj_paths['md'][model] = {}
            self.top_paths[model] = {}

            for replica in os.listdir(md_folder + '/output_models/' + model):
                replica_path = md_folder + '/output_models/' + model + '/' + replica
                if not os.path.isdir(replica_path):
                    continue

                # Define paths
                top_path = replica_path + '/em/prot_em.gro'
                nvt_path = replica_path + '/nvt/prot_nvt.xtc'
                npt_path = replica_path + '/npt'
                md_path = replica_path + '/md'

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

                if not os.path.exists(top_path) or not os.path.exists(nvt_path) or not os.path.exists(npt_path) or not os.path.exists(md_path):
                    continue

                # Get groups from topol index
                index_path = replica_path + '/topol/index.ndx'
                group_dics = _readGromacsIndexFile(index_path)
                if output_group not in group_dics:
                    raise ValueError(f'The selected output group is not available in the topol/index.ndx file. The following groups are available: {",".join(group_dics.keys())}')

                # Remove PBC
                centering_selector = group_dics['Protein']

                # Process topology file
                if not os.path.exists(top_path.replace('.gro', '_noPBC.gro')) or overwrite:
                    os.system(f'echo {centering_selector} {group_dics[output_group]} | {command} trjconv -s {top_path.replace(".gro", ".tpr")} -f {top_path} -o {top_path.replace(".gro", "_noPBC.gro")} -center -pbc res -ur compact -n {index_path}')
                    if remove_redundant_files:
                        os.remove(top_path)

                self.top_paths[model][replica] = top_path.replace('.gro', '_noPBC.gro')

                # Process NVT trajectory
                if not os.path.exists(nvt_path.replace('.xtc', '_noPBC.xtc')) or overwrite:
                    os.system(f'echo {centering_selector} {group_dics[output_group]} | {command} trjconv -s {nvt_path.replace(".xtc", ".tpr")} -f {nvt_path} -o {nvt_path.replace(".xtc", "_noPBC.xtc")} -center -pbc res -ur compact -n {index_path}')
                    if remove_redundant_files:
                        os.remove(nvt_path)

                self.traj_paths['nvt'][model][replica] = nvt_path.replace('.xtc', '_noPBC.xtc')

                # Process NPT trajectory
                if not os.path.exists(npt_path + '/prot_npt_cat_noPBC.xtc') or overwrite:
                    remove_paths = []
                    for file in os.listdir(npt_path):
                        if file.endswith('.xtc') and 'noPBC' not in file:
                            path = npt_path + '/' + file
                            os.system(f'echo {centering_selector} {group_dics[output_group]} | {command} trjconv -s {path.replace(".xtc", ".tpr")} -f {path} -o {path.replace(".xtc", "_noPBC.xtc")} -center -pbc res -ur compact -n {index_path}')
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
                            os.system(f'echo {centering_selector} {group_dics[output_group]} | {command} trjconv -s {path.replace(".xtc", ".tpr")} -f {path} -o {path.replace(".xtc", "_noPBC.xtc")} -center -pbc res -ur compact -n {index_path}')
                            remove_paths.append(path)


                    print(md_path)
                    # sort files in case they are higher than 10
                    file_list = [f for f in os.listdir(md_path) if f.endswith('_noPBC.xtc')]
                    sorted_file_list = sorted(file_list, key=lambda x: int(x.split('_')[2]))
                    sorted_file_list = ' '.join([md_path+'/'+f for f in sorted_file_list])

                    os.system(f'{command} trjcat -f {sorted_file_list} -o {md_path}/prot_md_cat_noPBC.xtc -cat')
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

        def angle(a, b, c):
            """
            Calculate the angle between three coordinate vectors
            """
            # Compute vectors BA and BC
            ba = a - b  # Vector from atom B to atom A
            bc = c - b  # Vector from atom B to atom C

            ba_norm = ba / np.linalg.norm(ba, axis=1)[:, np.newaxis]
            bc_norm = bc / np.linalg.norm(bc, axis=1)[:, np.newaxis]

            cos_theta = np.einsum('ij,ij->i', ba_norm, bc_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            angles = np.arccos(cos_theta)

            return np.rad2deg(angles)

        def euclidean(XA, XB, *, out=None):
            """Calculates the Euclidean distance between two points."""
            return np.sqrt(np.add.reduce(np.square(XA - XB), 1), out=out)


        for model in self.traj_paths[step]:
            if model not in self.distances:
                self.distances[model] = {}
                self.angles[model] = {}

            for replica in self.traj_paths[step][model]:
                if replica not in self.distances[model] or overwrite:
                    if not os.path.exists(self.traj_paths[step][model][replica]):
                        print(f'WARNING: trajectory file for model {model} and replica {replica} does not exist.')
                        continue
                    if not os.path.exists(self.top_paths[model][replica]):
                        print(f'WARNING: topology file for model {model} and replica {replica} does not exist.')
                        continue

                    traj = md.load(self.traj_paths[step][model][replica], top=self.top_paths[model][replica])
                    top = traj.topology

                    if replica not in self.angles[model]:
                        self.angles[model][replica] = {}

                    distance = {}
                    for m in metrics[model]:
                        joined_distances = []
                        for d in metrics[model][m]:
                            if len(d) == 2:
                                atom1 = top.select(f'resSeq {d[0][0]} and name {d[0][1]}')
                                atom2 = top.select(f'resSeq {d[1][0]} and name {d[1][1]}')

                                atom1_xyz = traj.xyz[:, atom1[0]]
                                atom2_xyz = traj.xyz[:, atom2[0]]
                                joined_distances.append(euclidean(atom1_xyz, atom2_xyz))

                                #joined_distances.append(md.compute_distances(traj,np.array([[atom1,atom2]])))

                            elif len(d) == 3:
                                atom1 = top.select(f'resSeq {d[0][0]} and name {d[0][1]}')
                                atom2 = top.select(f'resSeq {d[1][0]} and name {d[1][1]}')
                                atom3 = top.select(f'resSeq {d[2][0]} and name {d[2][1]}')

                                atom1_xyz = traj.xyz[:, atom1[0]]
                                atom2_xyz = traj.xyz[:, atom2[0]]
                                atom3_xyz = traj.xyz[:, atom3[0]]
                                self.angles[model][replica][m] = np.array(angle(atom1_xyz,atom2_xyz,atom3_xyz))

                                #self.angles[model] = md.compute_angles(traj,np.array([atom1,atom2,atom3]))
                            else:
                                raise ValueError('Metric has more than three atoms. This function can only be used to compute distances and angles.')

                        distance[m] = [min(values) for values in zip(*joined_distances)]

                    self.distances[model][replica] = np.array(distance)

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
                    command += 'cd ../../..\n'
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
            if model not in self.distances:
                self.distances[model] = {}
            for replica in self.top_paths[model]:
                dist_file = os.path.join(folder, model, str(replica), 'dist.json')
                if os.path.exists(dist_file):
                    with open(dist_file, 'r') as f:
                        distance_data = json.load(f)
                        # Convert list of lists to a single numpy array
                        for metric, values in distance_data.items():
                            distance_data[metric] = np.array(values).flatten()
                        self.distances[model][int(replica)] = distance_data
                    print(f'Reading distance for {model} - {replica}')
                else:
                    print(f'Warning: Distance file not found for {model} - {replica}')

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

    def get_distance_prob(self, threshold=0.45, group_metrics=None, combine_replicas=False):
        """
        Computes the probability of distances being below a threshold and groups metrics if provided.

        Parameters:
        threshold (float): Distance threshold for calculating probabilities (default: 0.45).
        group_metrics (dict): Dictionary defining groups of metrics for combined probability calculations (default: None).
        combine_replicas (bool): Whether to combine all replicas for each model (default: False).

        Returns:
        pd.DataFrame: DataFrame containing probabilities for each model and replica.

        Example:
        group_metrics = {
            'group1': ['metric1', 'metric2'],
            'group2': ['metric3', 'metric4']
        }
        This function calculates the probability that the distance for each metric (and group of metrics) is below the threshold.
        """
        data = {}  # Dictionary to store computed probabilities
        column_names = set()  # Set to store unique metric names

        if combine_replicas:
            # Iterate over each protein model
            for protein in self.distances:
                if not self.distances[protein]:
                    continue

                # Initialize combined_masks to accumulate data across replicas
                combined_masks = {metric: [] for metric in self.distances[protein][next(iter(self.distances[protein]))]}

                # Iterate over each replica
                for replica in self.distances[protein]:
                    # Iterate over each metric within the replica
                    for metric in self.distances[protein][replica]:
                        distances = self.distances[protein][replica][metric]
                        metric_mask = distances < threshold  # Boolean mask where distances are below threshold
                        combined_masks[metric].extend(metric_mask)  # Accumulate masks for combined analysis

                # Calculate probabilities for each metric
                for metric in combined_masks:
                    combined_mask = np.array(combined_masks[metric])
                    data[(protein, metric)] = np.mean(combined_mask)  # Mean probability of distances below threshold
                    column_names.add(metric)

                # Calculate probabilities for grouped metrics, if provided
                if group_metrics:
                    for group in group_metrics:
                        group_mask = np.ones(len(combined_masks[next(iter(combined_masks))]), dtype=bool)
                        for metric in combined_masks:
                            if metric in group_metrics[group]:
                                group_mask &= combined_masks[metric]  # Combine masks for the group

                        data[(protein, group)] = np.mean(group_mask)  # Mean probability for the group
                        column_names.add(group)
        else:
            # Iterate over each protein model
            for protein in self.distances:
                # Iterate over each replica
                for replica in self.distances[protein]:
                    # Iterate over each metric within the replica
                    for metric in self.distances[protein][replica]:
                        distances = self.distances[protein][replica][metric]
                        n_frames = len(distances)
                        metric_mask = distances < threshold  # Boolean mask where distances are below threshold

                        data[(protein, replica, metric)] = np.mean(metric_mask)  # Mean probability of distances below threshold
                        column_names.add(metric)

                    # Calculate probabilities for grouped metrics, if provided
                    if group_metrics:
                        for group in group_metrics:
                            group_mask = np.ones(n_frames, dtype=bool)
                            for metric in self.distances[protein][replica]:
                                if metric in group_metrics[group]:
                                    group_mask &= self.distances[protein][replica][metric] < threshold  # Combine masks for the group

                            data[(protein, replica, group)] = np.mean(group_mask)  # Mean probability for the group
                            column_names.add(group)

        # Prepare the DataFrame
        if combine_replicas:
            index = pd.MultiIndex.from_tuples([(protein, metric) for (protein, metric) in data.keys()], names=["Model", "Metric"])
        else:
            index = pd.MultiIndex.from_tuples([(protein, replica, metric) for (protein, replica, metric) in data.keys()], names=["Model", "Replica", "Metric"])

        # Create the DataFrame with probabilities
        self.df_prob = pd.DataFrame(list(data.values()), index=index, columns=["Probability"])

        # Unstack the DataFrame to have metrics as columns, filling missing values with 0
        self.df_prob = self.df_prob.unstack().fillna(0)

    def sort_prob_by_linear_combination(self, weights=None, keep_combined_score=False, filter_thresholds=None):
        """
        Calculates a combined score for each row based on the given weights, filters the DataFrame,
        stores the combined score in the DataFrame, and sorts the DataFrame by this score.

        Parameters:
        weights (dict): Dictionary specifying the weights for each metric. If None, equal weights are used.
        keep_combined_score (bool): Whether to keep the combined score column in the DataFrame after sorting (default: False).
        filter_thresholds (dict): Dictionary specifying the minimum thresholds for metrics to filter models (default: None).
        """
        probabilities = self.df_prob.copy()

        if filter_thresholds:
            probabilities = self.filter_by_thresholds(filter_thresholds, in_place=False)

        # If weights is None, assign equal weights to all metrics
        if weights is None:
            weights = {metric: 1/len(probabilities.columns) for metric in probabilities.columns}

        combined_score = sum(probabilities[metric] * weight for metric, weight in weights.items())
        probabilities['Combined_Score'] = combined_score
        probabilities = probabilities.sort_values(by='Combined_Score', ascending=False)

        if not keep_combined_score:
            probabilities = probabilities.drop(columns=['Combined_Score'])

        self.df_prob = probabilities

    def filter_by_thresholds(self, filter_thresholds, in_place=False):
        """
        Filters the DataFrame based on the specified thresholds.

        Parameters:
        filter_thresholds (dict): Dictionary specifying the minimum thresholds for metrics.
        in_place (bool): Whether to modify the DataFrame in place or return a new DataFrame (default: False).

        Returns:
        pd.DataFrame: The filtered DataFrame if in_place is False.
        """
        df = self.df_prob if in_place else self.df_prob.copy()

        if filter_thresholds:
            for metric, value in filter_thresholds.items():
                if ('Probability', metric) in df.columns:
                    df = df[df[('Probability', metric)] >= value]

        if not in_place:
            return df
        else:
            self.df_prob = df

    def plot_distance_prob(self, sort_by=None, filter_thresholds=None):
        """
        Plots the probability of distances being below a threshold for different models and replicas.

        Parameters:
        sort_by (str): Metric to sort the values by before plotting (default: None).
        filter_thresholds (dict): Dictionary specifying the minimum thresholds for metrics to filter models (default: None).
        """
        df_prob = self.df_prob.copy()

        # Apply filtering if specified
        if filter_thresholds:
            df_prob = self.filter_by_thresholds(filter_thresholds, in_place=False)

        # Check if the filtered DataFrame is empty
        if df_prob.empty:
            print("No data available for the specified filter thresholds.")
            return

        # Apply sorting if specified
        if sort_by and ('Probability', sort_by) in df_prob.columns:
            df_prob = df_prob.sort_values(by=('Probability', sort_by), ascending=False)

        # Ensure the sorted metric is always plotted first
        column_order = list(df_prob.columns)
        if sort_by and ('Probability', sort_by) in column_order:
            column_order.insert(0, column_order.pop(column_order.index(('Probability', sort_by))))

        num_metrics = len(column_order)
        colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown']  # Extend this list if you have more metrics
        fig, axes = plt.subplots(num_metrics, 1, figsize=(15, 4 * num_metrics), sharex=True)

        if num_metrics == 1:
            axes = [axes]  # Ensure axes is iterable even for a single subplot

        for ax, (metric, color) in zip(axes, zip(column_order, colors)):
            df_prob[metric].plot(kind='bar', ax=ax, width=0.8, color=color)
            ax.set_title(f'Probability of Distances Below Threshold for {metric}', fontsize=16)
            ax.set_ylabel('Probability', fontsize=14)
            ax.legend([metric], title='Metrics', fontsize=12)
            ax.grid(axis='y')

        axes[-1].set_xlabel('Model - Replica', fontsize=14)
        axes[-1].set_xticklabels(df_prob.index, rotation=45, ha='right', fontsize=10)

        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_distance_prob_scatter(self, x_col, y_col, color_col, filter_thresholds=None, weights=None, keep_combined_score=False):
        """
        Plots a scatter plot of probabilities with specified x, y, and color columns.

        Parameters:
        x_col (str): Column name for x-axis values.
        y_col (str): Column name for y-axis values.
        color_col (str): Column name for color values.
        filter_thresholds (dict): Dictionary specifying the minimum thresholds for metrics to filter models (default: None).
        weights (dict): Dictionary specifying the weights for each metric to calculate a combined score (default: None).
        keep_combined_score (bool): Whether to keep the combined score column in the DataFrame after sorting (default: False).
        """
        probabilities = self.df_prob['Probability'].copy()

        if filter_thresholds:
            filtered_df = self.filter_by_thresholds(filter_thresholds, in_place=False)
            probabilities = filtered_df['Probability']

        if weights:
            self.sort_prob_by_linear_combination(weights, keep_combined_score)
            probabilities = self.df_prob['Probability']

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(probabilities[x_col], probabilities[y_col], c=probabilities[color_col], cmap='viridis', s=100, alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(scatter, label='Probability of forming the ' + color_col + ' contact')
        plt.xlabel('Probability of forming the ' + x_col + ' contact', fontsize=14)
        plt.ylabel('Probability of forming the ' + y_col + ' contact', fontsize=14)
        plt.axvline(0.5, c='k', ls='--')
        plt.axhline(0.5, c='k', ls='--')
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.grid(True)
        plt.show()

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
