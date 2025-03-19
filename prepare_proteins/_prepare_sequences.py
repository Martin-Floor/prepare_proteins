from . import alignment
import os
import shutil
import bz2
import pickle
import pandas as pd
import mdtraj as md
import numpy as np
import json
import subprocess
import io
from pkg_resources import Requirement, resource_listdir, resource_stream
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import seaborn as sns

class sequenceModels:

    def __init__(self, sequences_fasta):

        if isinstance(sequences_fasta, str):
            self.sequences = alignment.readFastaFile(sequences_fasta, replace_slash=True)
        elif isinstance(sequences_fasta, dict):
            self.sequences = sequences_fasta
        else:
            raise ValueError('sequences_fasta must be a string or a dictionary containing the sequences!')

        self.sequences_names = list(self.sequences.keys())

    def setUpAlphaFold(self, job_folder, model_preset='monomer_ptm', exclude_finished=True,
                       remove_extras=False, remove_msas=False, only_models=None, gpu_relax=True):
        """
        Set up AlphaFold predictions for the loaded sequneces
        """

        if isinstance(only_models, str):
            only_models = [only_models]

        # Create Job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if not os.path.exists(job_folder+'/input_sequences'):
            os.mkdir(job_folder+'/input_sequences')

        if not os.path.exists(job_folder+'/output_models'):
            os.mkdir(job_folder+'/output_models')

        # Check for finished models
        excluded = []
        if exclude_finished:
            for model in os.listdir(job_folder+'/output_models'):

                if not isinstance(only_models, type(None)):
                    if model not in only_models:
                        continue

                for f in os.listdir(job_folder+'/output_models/'+model):
                    if f == 'ranked_0.pdb' or f == 'ranked__0.pdb.bz2':
                        excluded.append(model)

        jobs = []
        for model in self.sequences:

            if not isinstance(only_models, type(None)):
                if model not in only_models:
                    continue

            if exclude_finished and model in excluded:
                continue

            sequence = {}
            sequence[model] = self.sequences[model]
            alignment.writeFastaFile(sequence, job_folder+'/input_sequences/'+model+'.fasta')
            command = 'cd '+job_folder+'\n'
            command += 'Path=$(pwd)\n'
            command += 'bsc_alphafold --fasta_paths $Path/input_sequences/'+model+'.fasta'
            command += ' --output_dir=$Path/output_models'
            command += ' --model_preset='+model_preset
            command += ' --max_template_date=2022-01-01'
            if gpu_relax:
                command += ' --use_gpu_relax=True'
            else:
                command += ' --use_gpu_relax=False'
            command += ' --random_seed 1\n'
            if remove_extras:
                command += f'rm -r $Path/output_models/{model}/msas\n'
                command += f'rm -r $Path/output_models/{model}/*.pkl\n'

            if remove_msas:
                command += f'rm -r $Path/output_models/{model}/msas\n'

            command += 'cd ..\n'

            jobs.append(command)

        return jobs

    def copyModelsFromAlphaFoldCalculation(self, af_folder, output_folder, prefix='',
                                           return_missing=False, copy_all=False):
        """
        Copy models from an AlphaFold calculation to an specfied output folder.

        Parameters
        ==========
        af_folder : str
            Path to the Alpha fold folder calculation.
        output_folder : str
            Path to the output folder where to store the models.
        return_missing : bool
            Return a list with the missing models.
        """

        # Get paths to models in alphafold folder
        models_paths = {}
        for d in os.listdir(af_folder+'/output_models'):
            mdir = af_folder+'/output_models/'+d
            for f in os.listdir(mdir):
                if f.startswith('relaxed_model_1_ptm'):
                    models_paths[d] = mdir+'/'+f
                elif f.startswith('ranked_0'):
                    models_paths[d] = mdir+'/'+f

        # Create output folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if return_missing:
            missing = []

        af_models = []
        if copy_all:
            af_models = list(models_paths.keys())
        else:
            for m in self:
                if m in models_paths:
                    af_models.append(m)
                elif return_missing:
                    missing.append(m)
                    print('Alphafold model for sequence %s was not found in folder %s' % (m, af_folder))

        for m in af_models:
            if models_paths[m].endswith('.pdb'):
                shutil.copyfile(models_paths[m], output_folder+'/'+prefix+m+'.pdb')
            elif models_paths[m].endswith('.bz2'):
                file = bz2.BZ2File(models_paths[m], 'rb')
                pdbfile = open(output_folder+'/'+prefix+m+'.pdb', 'wb')
                shutil.copyfileobj(file, pdbfile)

        if return_missing:
            return missing

    def loadAFScores(self, af_folder, only_indexes=None):
        """
        Read AF scores from the models AF-pkl files.

        Parameters
        ==========
        af_folder : str
            Path to the AF calculation folder.
        only_indexes : list
            Only extract scores for the models with the given index.

        #TODO Only PTM is implemented. Then implement extraction of other scores...
        """
        if only_indexes:
            if isinstance(only_indexes, int):
                only_indexes = [only_indexes]
            if not isinstance(only_indexes, list):
                raise ValueError('only_indexes must be a list or a integer!')

        af_data = {}
        af_data['Model'] = []
        af_data['Index'] = []
        af_data['ptm'] = []

        for f in sorted(os.listdir(af_folder+'/output_models')):
            model = f
            model_folder = af_folder+'/output_models/'+f
            if not os.path.isdir(model_folder):
                continue

            for g in sorted(os.listdir(model_folder)):
                if not g.endswith('.pkl') or not g.startswith('result_model'):
                    continue

                index = int(g.split('_')[2])

                if only_indexes and index not in only_indexes:
                    continue

                with open(model_folder+'/'+g, "rb") as pkl:
                    pkl_object = pickle.load(pkl)

                af_data['Model'].append(model)
                af_data['Index'].append(index)
                af_data['ptm'].append(pkl_object['ptm'])

        af_data = pd.DataFrame(af_data).set_index(['Model', 'Index'])

        return af_data

    def mutate_fasta(self,model,mut_position=int,wt_res=str,new_res=str,start_position=1):

        """
            Mutates a protein sequence based and updates the sequences dictionary.

            To avoid runtime error, **import copy** and make a copy of self.sequences before iterating the sequences.

            Parameters:
            - model: A string representing the model identifier.
            - mut_position: An integer representing the position in the sequence to be mutated.
            - wt_res: A string representing the wild-type (original) residue at the mutation position.
            - new_res: A string representing the new residue to replace the wild-type residue.
            - start_position: An integer representing the starting position of the sequence (default is 1).

            Raises:
            - AssertionError: If the wild-type residue at the specified position does not match the given wt_res.
            - AssertionError: If the mutation operation does not result in the expected new_res at the mutated position.


        """

        seq = self.sequences[model]
        pos = mut_position - start_position

        assert(seq[pos] == wt_res)

        mutant_seq = seq[:pos]+new_res+seq[pos+1:]

        assert(mutant_seq[pos] == new_res)
        assert(len(seq) == len(mutant_seq))

        new_seq_id = model+"_"+wt_res+str(mut_position)+new_res

        self.sequences[new_seq_id] = mutant_seq

        self.sequences_names.append(new_seq_id)

        return

    def setUpBioEmu(self, job_folder, num_samples=10000, batch_size_100=20, gpu_local=False,
                    verbose=True, models=None, skip_finished=False,
                    bioemu_env=None, conda_sh='~/miniconda3/etc/profile.d/conda.sh'):
        """
        Set up and optionally execute BioEmu commands for each model sequence.

        For each model in self.sequences, this function creates a dedicated folder structure
        (job folder, model folder, and cache directory). If a Conda environment is provided
        via the 'bioemu_env' parameter, the function activates the specified Conda environment,
        runs a sample command with minimal sample settings, and deactivates the environment
        (this is relevant if your computing node does not have access to internet, e.g., MN5).
        Otherwise, it compiles and returns a list of command strings (which can be executed later) for each model.

        Parameters:
            job_folder (str): The root folder where job outputs will be stored.
            num_samples (int): Number of samples to run in BioEmu (default 10000).
            batch_size_100 (int): Batch size (default 20).
            gpu_local (bool): If True, sets the CUDA_VISIBLE_DEVICES variable for running GPUs in
                              local computer with multiple GPUS.
            bioemu_env (str): Name of the Conda environment to activate. If provided, the command
                              is executed to store the cached files that are obtained with an internet connection.
            conda_sh (str): Path to the conda.sh initialization script (default '~/miniconda3/etc/profile.d/conda.sh').

        Returns:
            Returns a list of command strings to run bioemu for each model.
        """
        import os
        import subprocess

        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        jobs = []
        for model in self.sequences:

            if models and model not in models:
                continue

            model_folder = os.path.join(job_folder, model)
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)

            if skip_finished:

                sample_file = os.path.join(model_folder, 'samples.xtc')
                topology_file = os.path.join(model_folder, 'topology.pdb')

                if os.path.exists(sample_file) and os.path.exists(topology_file):
                    traj = md.load(sample_file, top=topology_file)

                    if traj.n_frames >= num_samples:
                        print(f'{model} has already sampled {num_samples} poses. Skipping it.')
                        continue

            cache_embeds_dir = os.path.join(model_folder, 'cache')
            if not os.path.exists(cache_embeds_dir):
                os.mkdir(cache_embeds_dir)

            if bioemu_env:

                cached_files = [f for f in os.listdir(cache_embeds_dir)]
                fasta_cached_file = [f for f in cached_files if f.endswith('.fasta')]
                npy_cached_files = [f for f in cached_files if f.endswith('.npy')]

                if len(fasta_cached_file) == 1 and len(npy_cached_files) == 2:
                    if verbose:
                        print(f'Input files for model {model} were found.')
                else:
                    command = f"""
                    source {conda_sh}
                    conda activate {bioemu_env}
                    python -m bioemu.sample --sequence {self.sequences[model]} --num_samples 1 --batch_size_100 {batch_size_100} --cache_embeds_dir {cache_embeds_dir} --output_dir {model_folder}
                    conda deactivate
                    """
                    if verbose:
                        print(f"Setting input files for model {model}")
                    result = subprocess.run(["bash", "-i", "-c", command], capture_output=True, text=True)

            command = 'RUN_SAMPLES='+str(num_samples)+'\n'
            command += 'while true; do\n'
            #command += 'FILE_COUNT=$(find "'+job_folder+'/'+model+'/batch*'+'" -type f | wc -l)\n'
            if gpu_local:
                command += 'CUDA_VISIBLE_DEVICES=GPUID '
            command += 'python -m bioemu.sample '
            command += f'--sequence {self.sequences[model]} '
            command += f'--num_samples $RUN_SAMPLES '
            command += f'--batch_size_100 {batch_size_100} '
            command += f'--cache_embeds_dir {cache_embeds_dir} '
            command += f'--output_dir {model_folder}\n'
            command += 'NUM_SAMPLES=$(python -c "import mdtraj as md; traj = md.load_xtc(\''+job_folder+'/'+model+'/samples.xtc\', top=\''+job_folder+'/'+model+'/topology.pdb\'); print(traj.n_frames)")\n'
            command += 'if [ "$NUM_SAMPLES" -ge '+str(num_samples)+' ]; then\n'
            command += 'echo "All samples computed. Exiting."\n'
            command += 'exit 0\n'
            command += 'fi\n'
            command += 'RUN_SAMPLES=$(($RUN_SAMPLES+'+str(num_samples)+'-$NUM_SAMPLES))\n'
            command += 'done \n'

            jobs.append(command)

        return jobs

    def clusterBioEmuSamples(self, job_folder, bioemu_folder, models=None, stderr=True, stdout=True,
                             output_dcd=False, output_pdb=False, c=0.9, cov_mode=0, verbose=True,
                             evalue=10.0, overwrite=False, remove_input_pdb=True, min_sampled_points=None):
        """
        Process BioEmu models by extracting trajectory frames (saved as PDBs) and clustering them.
        Each model’s clustering results are cached in its model folder (clusters.json) and then
        compiled into an overall cache file (overall_clusters.json) in the job folder. Optionally,
        after clustering the extracted input PDBs are removed.

        The model name from the bioemu folder is used as the prefix for naming output PDB files.

        Parameters
        ----------
        job_folder : str
            Path to the job folder where model subfolders will be created.
        bioemu_folder : str
            Path to the folder containing BioEmu models.
        models : list, optional
            List of model names to process. If None, all models in bioemu_folder will be processed.
        output_dcd : bool, optional
            If True, save clustered structures as DCD files (default: False).
        output_pdb : bool, optional
            If True, save clustered structures as PDB files (default: False).
        c : float, optional
            Fraction of aligned residues for clustering (default: 0.9).
        cov_mode : int, optional
            Coverage mode for clustering (default: 0).
        evalue : float, optional
            E-value threshold for clustering (default: 10.0).
        overwrite : bool, optional
            If True, previous clustering outputs and caches are deleted (default: False).
        remove_input_pdb : bool, optional
            If True, remove the extracted input PDB files (input_models folder) after clustering.
            Default is True.

        Returns
        -------
        overall_clusters : dict
            Dictionary mapping each model to its clustering output.
        """

        # Convert paths to absolute.
        job_folder = os.path.abspath(job_folder)
        bioemu_folder = os.path.abspath(bioemu_folder)

        # Define overall cache file.
        overall_cache_file = os.path.join(job_folder, "overall_clusters.json")

        # Load existing overall clusters if not overwriting
        if os.path.exists(overall_cache_file) and not overwrite:
            if verbose:
                print("Loading cached overall clusters from", overall_cache_file)
            with open(overall_cache_file, "r") as f:
                overall_clusters = json.load(f)
        else:
            overall_clusters = {}

        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        # Loop over each model in the bioemu_folder.
        for model in os.listdir(bioemu_folder):
            if models and model not in models:
                continue

            # Skip models that are already in the cache unless overwrite=True
            if model in overall_clusters and not overwrite:
                if verbose:
                    print(f"Skipping model {model}, already clustered and cached.")
                continue

            # Delete previous clustering information
            model_folder = os.path.join(job_folder, model)
            if overwrite and os.path.exists(model_folder):
                for item in os.listdir(model_folder):
                    item_path = os.path.join(model_folder, item)
                    if item == 'input_models':
                        if remove_input_pdb and os.path.exists(item_path):
                            shutil.rmtree(item_path)
                    else:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)


            if not os.path.exists(model_folder):
                os.mkdir(model_folder)

            input_models_folder = os.path.join(model_folder, 'input_models')
            if not os.path.exists(input_models_folder):
                os.mkdir(input_models_folder)

            # Define file paths for topology and trajectory.
            top_file = os.path.join(bioemu_folder, model, 'topology.pdb')
            traj_file = os.path.join(bioemu_folder, model, 'samples.xtc')

            if not os.path.exists(top_file):
                print(f'WARNING: No topology file was found for model {model}. Skipping it.')
                continue

            if not os.path.exists(traj_file):
                print(f'WARNING: No trajectory file was found for model {model}. Skipping it.')
                continue

            # Load the topology and trajectory; superpose the trajectory.
            traj_top = md.load(top_file)
            traj = md.load(traj_file, top=top_file)

            if min_sampled_points and traj.n_frames < min_sampled_points:
                print(f'WARNING: trajectory file for {model} has only {traj.n_frames} poses. Skipping it.')
                continue

            traj.superpose(traj_top[0])

            # Save each frame of the trajectory as a separate PDB file.
            for i in range(traj.n_frames):
                pdb_path = os.path.join(input_models_folder, f"frame_{i:07d}.pdb")
                if not os.path.exists(pdb_path):
                    traj[i].save_pdb(pdb_path)

            # Use the model name as the prefix.
            prefix = model

            # Run structural clustering on the extracted PDBs.
            if verbose:
                print(f"Clustering PDBs for model {model}...")
            clusters = _structuralClustering(
                job_folder=model_folder,
                models_folder=input_models_folder,
                output_dcd=output_dcd,
                save_as_pdb=output_pdb,
                model_prefix=prefix,
                c=c,
                cov_mode=cov_mode,
                evalue=evalue,
                overwrite=overwrite,
                stderr=stderr,
                stdout=stdout,
                verbose=verbose
            )
            overall_clusters[model] = clusters
            if verbose:
                print(f"Clusters for model {model}: {clusters}")

            # Optionally remove the extracted PDB files.
            if remove_input_pdb and os.path.exists(input_models_folder):
                shutil.rmtree(input_models_folder)
                print(f"Removed input PDB folder: {input_models_folder}")

        # Cache the updated overall clusters.
        with open(overall_cache_file, "w") as f:
            json.dump(overall_clusters, f, indent=2)
        if verbose:
            print("Overall clusters cached to", overall_cache_file)

        return overall_clusters

    def setUpBioEmuClustering(self, job_folder, bioemu_folder, evalues, sensitivity=7.5 , models=None, c=0.9, cov_mode=0,
                              overwrite=False, min_sampled_points=None, skip_finished=True, verbose=True,
                              cluster_reassign=True):
        """
        Set up clustering jobs by copying a Python script into the job folder, extracting trajectory frames,
        and generating execution commands.

        Parameters
        ----------
        job_folder : str
            Path to the job folder where model subfolders will be created.
        bioemu_folder : str
            Path to the folder containing BioEmu models.
        evalues : float or list
            List of E-value thresholds for clustering. Each value will have a unique subfolder.
        models : list, optional
            List of model names to process. If None, all models in bioemu_folder will be processed.
        c : float, optional
            Fraction of aligned residues for clustering (default: 0.9).
        cov_mode : int, optional
            Coverage mode for clustering (default: 0).
        overwrite : bool, optional
            If True, previous clustering outputs and caches are deleted (default: False).
        min_sampled_points : int, optional
            Minimum number of sampled trajectory frames required for clustering.
        skip_finished : bool, optional
            If True, skip generating commands for model–evalue combinations where the results file
            already exists (default: True).
        verbose : bool, optional
            If True, print progress messages.

        Returns
        -------
        jobs : list
            List of command strings for executing clustering.
        """

        def format_evalue(x):
            """
            Format a float scientific value (e.g., 1e-30 or 0.5e-30)
            into a string of the form "e_0.5e-30" or "e_1.0e-30".
            """
            # x is the actual evalue, e.g., 1e-30
            # Compute the exponent: for 1e-30, log10(x) is -30.
            exponent = -int(round(np.log10(x)))
            # Compute the coefficient: x * 10**exponent gives 1.0 or 0.5.
            coefficient = x * (10 ** exponent)
            return f"e_{coefficient:.1f}e-{exponent}"

        # Convert single evalue to a list if necessary
        if isinstance(evalues, float):
            evalues = [evalues]

        # Use relative paths
        job_folder = os.path.relpath(job_folder)
        bioemu_folder = os.path.relpath(bioemu_folder)

        os.makedirs(job_folder, exist_ok=True)

        # Copy the clustering Python script to the job folder.
        # Assumes that _copyScriptFile copies "foldseekClustering.py" from a known location.
        _copyScriptFile(job_folder, "foldseekClustering.py")

        jobs = []
        for model in os.listdir(bioemu_folder):
            if models and model not in models:
                continue

            model_path = os.path.join(bioemu_folder, model)
            model_folder = os.path.join(job_folder, model)
            os.makedirs(model_folder, exist_ok=True)

            # Extraction step: create input_models folder and extract frames if needed.
            input_models_folder = os.path.join(model_folder, 'input_models')
            os.makedirs(input_models_folder, exist_ok=True)

            # Locate topology and trajectory files in the BioEmu model folder.
            top_file = os.path.join(model_path, 'topology.pdb')
            traj_file = os.path.join(model_path, 'samples.xtc')
            if not os.path.exists(top_file):
                print(f"WARNING: No topology file found for model {model}. Skipping it.")
                continue
            if not os.path.exists(traj_file):
                print(f"WARNING: No trajectory file found for model {model}. Skipping it.")
                continue

            # Load the trajectory using mdtraj.
            traj_top = md.load(top_file)
            traj = md.load(traj_file, top=top_file)

            # If a minimum number of frames is specified, check against trajectory length.
            if min_sampled_points and traj.n_frames < min_sampled_points:
                print(f"WARNING: Model {model} has only {traj.n_frames} frames (min required is {min_sampled_points}). Skipping.")
                continue

            # Check if the number of extracted PDB files equals the number of frames.
            existing_pdbs = [f for f in os.listdir(input_models_folder) if f.endswith('.pdb')]
            if len(existing_pdbs) != traj.n_frames:
                if verbose:
                    print(f"Extracting {traj.n_frames} frames for model {model}...")
                # Superpose on the first frame
                traj.superpose(traj_top[0])
                for i in range(traj.n_frames):
                    pdb_path = os.path.join(input_models_folder, f"frame_{i:07d}.pdb")
                    traj[i].save_pdb(pdb_path)
                if verbose:
                    print(f"Saved {traj.n_frames} frames for model {model}.")
            else:
                if verbose:
                    print(f"Skipping extraction for model {model} (found {len(existing_pdbs)} PDBs).")

            # Generate clustering commands for each evalue.
            for evalue in evalues:
                # Create a shorter folder name for the evalue (e.g., 1e-3 becomes e_1e-3)
                formatted_evalue = format_evalue(float(evalue))
                evalue_folder = os.path.join(model_folder, formatted_evalue)
                os.makedirs(evalue_folder, exist_ok=True)

                # Check for existing clustering results if skip_finished is set.
                clusters_file = os.path.join(evalue_folder, "result", "clusters.json")
                if skip_finished and os.path.exists(clusters_file):
                    if verbose:
                        print(f"Skipping clustering for model {model}, evalue {evalue} "
                              f"(results already exist in {clusters_file}).")
                    continue

                # Generate command using relative paths.
                # From evalue_folder, the foldseekClustering.py script is at ../../foldseekClustering.py,
                # and the input_models folder is at ../../input_models.
                command = f"cd {evalue_folder}\n"
                command += f"python ../../._foldseekClustering.py ../input_models result tmp "
                command += f"--cov_mode {cov_mode} --evalue {evalue} --c {c}  --sensitivity {sensitivity} "
                if cluster_reassign:
                    command += f"--cluster-reassign"
                command += '\n'
                command += f"cd ../../..\n"
                jobs.append(command)

                if verbose:
                    print(f"Prepared command for model {model}, evalue {evalue} → Folder: {formatted_evalue}")

        return jobs

    def readBioEmuClusteringResults(self, job_folder, models=None, verbose=True):
        """
        Reads clustering results from a job folder and returns a dictionary mapping each model
        and e-value folder to its clustering results.

        The expected folder structure is:

            job_folder/
            ├── model1/
            │   ├── e_1e-04/
            │   │   └── result/
            │   │       └── clusters.json
            │   ├── e_1e-03/
            │   │   └── result/
            │   │       └── clusters.json
            │   └── ...
            ├── model2/
            │   └── e_1e-04/
            │       └── result/
            │           └── clusters.json
            └── ...

        Parameters
        ----------
        job_folder : str
            Path to the job folder containing model subfolders.
        models : list, optional
            List of model names to process. If None, all subdirectories in job_folder are processed.
        verbose : bool, optional
            If True, prints progress messages.

        Returns
        -------
        results : dict
            Dictionary in the form:
            {
                "model1": {
                    "e_1e-04": { ... clustering results ... },
                    "e_1e-03": { ... clustering results ... }
                },
                "model2": {
                    "e_1e-04": { ... clustering results ... },
                    ...
                },
                ...
            }
        """

        results = {}
        for model in os.listdir(job_folder):
            model_path = os.path.join(job_folder, model)
            if not os.path.isdir(model_path):
                continue
            if models and model not in models:
                continue

            results[model] = {}
            # Loop over each folder in the model directory; we assume e-value folders start with "e_"
            for item in os.listdir(model_path):
                if not item.startswith("e_"):
                    continue
                evalue_folder = os.path.join(model_path, item)
                result_dir = os.path.join(evalue_folder, "result")
                clusters_file = os.path.join(result_dir, "clusters.json")
                if os.path.exists(clusters_file):
                    try:
                        with open(clusters_file, "r") as f:
                            clusters = json.load(f)
                        results[model][item] = clusters
                        if verbose:
                            print(f"Read clustering results for model '{model}', folder '{item}'.")
                    except Exception as e:
                        if verbose:
                            print(f"Error reading {clusters_file}: {e}")
                else:
                    if verbose:
                        print(f"Warning: {clusters_file} not found for model '{model}', folder '{item}'.")

        return results

    def fitBioEmuClusteringToHillEquation(self, clustering_data, plot=True, plot_fits_only=False, verbose=False,
                                          plot_only=None, plot_legend=True):
        """
        Analyze clustering data from readBioEmuClusteringResults.

        Parameters:
        - clustering_data (dict): Output from readBioEmuClusteringResults.
          Expected to be a dictionary structured as {model: {folder: clusters, ...}, ...}.
        - plot (bool): Optional flag to generate and show plots. Default is True.

        Returns:
        - model_half_evalues (dict): Fitted E_half values for each model.
        - model_slopes (dict): Fitted Hill slopes for each model.
        """

        def hill_equation(e, y_min, y_max, e_half, n):
            return y_min + (y_max - y_min) / (1 + (e_half / e)**n)

        # Dictionaries to store fitted E_half values and Hill slopes for each model
        model_half_evalues = {}
        model_slopes = {}

        # Extract and sort data
        data = {}
        for model, evalue_dict in clustering_data.items():
            e_vals, counts = [], []
            for folder, clusters in evalue_dict.items():
                try:
                    e_val = float(folder[2:])  # Extract numeric e-value from folder name
                    e_vals.append(e_val)
                    counts.append(len(clusters))
                except ValueError:
                    continue
            if e_vals:
                idx = np.argsort(e_vals)
                data[model] = (np.array(e_vals)[idx], np.array(counts)[idx])

        if plot:
            # Create the first plot: E-value vs. Number of Clusters with Hill Fit
            fig, ax = plt.subplots(figsize=(8, 6))

            # Handles for the first legend
            data_handle = Line2D([], [], color='black', marker='o', linestyle='-', label='Data')
            fit_handle  = Line2D([], [], color='black', linestyle='--', label='Fit')

            # Handles for the model-specific legend
            model_handles = []
            colors = plt.cm.tab20(np.linspace(0, 1, len(data)))

        # Loop through each model and perform the fit
        for i, (model, (e_vals, clusters)) in enumerate(data.items()):
            if plot:
                color = colors[i % len(colors)]

                if plot_only and model not in plot_only:
                    continue

                if not plot_fits_only:
                    # Plot the data
                    ax.plot(e_vals, clusters, marker='o', linestyle='-', color=color, alpha=0.8)
                    # Create a model legend handle
                    model_handle = Line2D([], [], color=color, marker='o', linestyle='-', label=model)
                    model_handles.append(model_handle)
            else:
                color = 'blue'  # Default color when not plotting

            # Fit the data if there are enough points
            if len(e_vals) >= 4:

                slope_guess = 0.5
                y_min_guess = clusters.min()
                y_max_guess = clusters.max()
                half_y = y_min_guess + (y_max_guess - y_min_guess) / 2

                # Convert to log10 once (for interpolation)
                log_e = np.log10(e_vals)

                # Boolean mask: True for points ≥ half‑maximum
                above = clusters >= half_y

                if np.any(above) and np.any(~above):
                    # Last index where cluster is still above half_y
                    i = np.where(above)[0][-1]
                    # Guard: if it’s not the very last point, interpolate
                    if i < len(e_vals) - 1:
                        y1, y2 = clusters[i], clusters[i+1]
                        x1, x2 = log_e[i], log_e[i+1]
                        frac = (y1 - half_y) / (y1 - y2)
                        log10_e_half_guess = x1 + frac * (x2 - x1)
                        e_half_guess = 10**log10_e_half_guess
                    else:
                        e_half_guess = e_vals[i]
                else:
                    # No crossing (flat region at one end) → geometric median
                    e_half_guess = 10**np.median(log_e)

                try:
                    popt, _ = curve_fit(
                        hill_equation,
                        e_vals,
                        clusters,
                        p0=[y_min_guess, y_max_guess, e_half_guess, slope_guess]
                    )
                    y_min_fit, y_max_fit, e_half_fit, n_fit = popt

                    # Store the fitted values
                    model_half_evalues[model] = e_half_fit
                    model_slopes[model] = n_fit

                    if plot:

                        if plot_only and model not in plot_only:
                            continue

                        # Generate a smooth range for the fit curve
                        e_fine = np.logspace(np.log10(e_vals.min()), np.log10(e_vals.max()), 300)
                        fit_curve = hill_equation(e_fine, y_min_fit, y_max_fit, e_half_fit, n_fit)
                        # Plot the fit as a dashed line
                        ax.plot(e_fine, fit_curve, '--', color=color, alpha=0.8)

                except Exception as ex:
                    print(f"Error fitting {model}: {ex}")
            else:
                print(f"Skipping fit for {model}: only {len(e_vals)} data points")

        if plot:
            # Configure the first plot
            ax.set_xscale('log')
            ax.set_xlabel("E-value", fontsize=12)
            ax.set_ylabel("Number of Clusters", fontsize=12)
            ax.set_title("E-value vs. Number of Clusters (Hill Fit)", fontsize=14, fontweight='bold')

            # Add legends
            if plot_legend:
                legend1 = ax.legend(handles=[data_handle, fit_handle], loc='upper left', title="Plot Types")
                ax.add_artist(legend1)
                legend2 = ax.legend(handles=model_handles, loc='lower left', bbox_to_anchor=(0.02, 0.02),
                                    title="Models", ncol=1, frameon=False)
                ax.add_artist(legend2)

            plt.subplots_adjust(left=0.2, bottom=0.15)
            plt.show()

        # Optionally, print out the fitted values for confirmation
        if verbose:
            print("Fitted E_half values and Hill slopes:")
            for m in model_half_evalues:
                print(f"  {m}: E_half = {model_half_evalues[m]:.4g}, slope = {model_slopes[m]:.4g}")

        return model_half_evalues, model_slopes

    def computeBioEmuRMSF(self, bioemu_folder, ref_pdb, plot=False, ylim=None):
        """
        Computes RMSF values for all models in the specified folder and optionally plots RMSF by residue.

        Parameters:
        - bioemu_folder (str): Path to the folder containing model subdirectories with trajectory and topology files.
        - ref_pdb (str): Path to the reference PDB structure for RMSF calculation.
        - plot (bool, optional): Whether to generate a plot of RMSF vs. residue. Default is False.

        Returns:
        - rmsf (dict): Dictionary containing RMSF arrays for each model.
          Each array holds the per-residue RMSF (in nm) for the selected backbone (Cα) atoms.
        """
        # Load reference structure and select backbone Cα atoms
        if isinstance(ref_pdb,str):
            ref = md.load(ref_pdb)
            ref_bb_atoms = ref.topology.select('name CA')

        # Dictionary to store RMSF values for each model
        rmsf = {}

        # Iterate through each model folder
        for model in os.listdir(bioemu_folder):

            # Load reference structure and select backbone Cα atoms
            if isinstance(ref_pdb,dict):
                ref = md.load(ref_pdb[model])
                ref_bb_atoms = ref.topology.select('name CA')

            traj_file = f'{bioemu_folder}/{model}/samples.xtc'
            top_file = f'{bioemu_folder}/{model}/topology.pdb'

            if not os.path.exists(traj_file):
                continue

            traj = md.load(traj_file, top=top_file)
            traj_bb_atoms = traj.topology.select('name CA')

            # Compute RMSF for the selected atoms (per-residue fluctuations)
            rmsf[model] = md.rmsf(traj, ref, atom_indices=traj_bb_atoms)

        if plot:
            # Get the residue numbers corresponding to the selected Cα atoms from the reference structure
            residue_ids = [ref.topology.atom(i).residue.resSeq for i in ref_bb_atoms]

            plt.figure(figsize=(12, 6))
            # Plot each model's RMSF as a line plot
            for model, rmsf_values in rmsf.items():
                plt.plot(residue_ids, rmsf_values, label=model)

            if ylim:
                plt.ylim(ylim)
            plt.xlabel("Residue Number", fontsize=12)
            plt.ylabel("RMSF (nm)", fontsize=12)
            plt.title("RMSF by Residue", fontsize=14, fontweight='bold')
            plt.legend()
            plt.tight_layout()

        return rmsf

    def computeBioEmuRMSD(self, bioemu_folder, ref_pdb, plot=False):
        """
        Computes RMSD values for all models in the specified folder and optionally plots a violin plot.

        Parameters:
        - bioemu_folder (str): Path to the folder containing model subdirectories with trajectory and topology files.
        - ref_pdb (str): Path to the reference PDB structure for RMSD calculation.
        - plot (bool, optional): Whether to generate a violin plot. Default is True.

        Returns:
        - rmsd (dict): Dictionary containing RMSD arrays for each model.
        """

        if isinstance(ref_pdb,str):
            # Load reference structure
            ref = md.load(ref_pdb)
            ref_bb_atoms = ref.topology.select('name CA')

        # Dictionary to store RMSD values
        rmsd = {}

        # Iterate through each model folder
        for model in os.listdir(bioemu_folder):

            if isinstance(ref_pdb,dict):
                ref = md.load(ref_pdb[model])
                ref_bb_atoms = ref.topology.select('name CA')

            traj_file = f'{bioemu_folder}/{model}/samples.xtc'
            top_file = f'{bioemu_folder}/{model}/topology.pdb'

            if not os.path.exists(traj_file):
                continue

            traj = md.load(traj_file, top=top_file)
            traj_bb_atoms = traj.topology.select('name CA')

            rmsd[model] = md.rmsd(traj, ref, atom_indices=traj_bb_atoms, ref_atom_indices=ref_bb_atoms)

        if plot:
            # Compute average RMSD for each model and order models by average value
            model_avg = {model: np.mean(vals) for model, vals in rmsd.items()}
            ordered_models = sorted(model_avg, key=model_avg.get)

            # Build a DataFrame for plotting
            data = []
            for model, values in rmsd.items():
                for value in values:
                    data.append({'Model': model, 'RMSD': value})
            df = pd.DataFrame(data)

            # Create a violin plot using seaborn
            plt.figure(figsize=(10, 6))
            sns.violinplot(x='Model', y='RMSD', data=df, order=ordered_models, inner="quartile")

            plt.xticks(rotation=45)
            plt.title("RMSD Distributions Ordered by Average RMSD")
            plt.tight_layout()
            plt.show()

        return rmsd

    def setUpInterProScan(self, job_folder, not_exclude=['Gene3D'], output_format='tsv',
                          cpus=40, version="5.71-102.0", max_bin_size=10000):
        """
        Set up InterProScan analysis to search for domains in a set of proteins

        not_exclude: list
            refers to programs that will used to find the domains, and that will retrieve results. This has an
            impact on the time.

        If an update is needed, download in bubbles the new version (replace XX-XX with the version number):
            wget -O interproscan-5.XX-XX.0-64-bit.tar.gz http://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.XX-XX.0/interproscan-5.XX-XX.0-64-bit.tar.gz

        To get the link you can also visit:
            https://www.ebi.ac.uk/interpro/about/interproscan/
        """

        if isinstance(not_exclude, str):
            not_exclude = [not_exclude]

        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if not os.path.exists(job_folder+'/input_fasta'):
            os.mkdir(job_folder+'/input_fasta')

        if not os.path.exists(job_folder+'/output'):
            os.mkdir(job_folder+'/output')

        # Define number of bins to execute interproscan
        n_bins = len(self.sequences) // max_bin_size
        if len(self.sequences) % 10000 > 0:
            n_bins += 1
        zf = len(str(n_bins))

        # Get all sequences names
        all_sequences = list(self.sequences.keys())

        # Create commands for each sequence bin
        jobs = []
        for i in range(n_bins):

            bin_index = str(i).zfill(zf)

            input_file = 'input_fasta/input_'+bin_index+'.fasta'

            bin_sequences = {s:self.sequences[s] for s in all_sequences[i*max_bin_size:(i+1)*max_bin_size]}

            alignment.writeFastaFile(bin_sequences, job_folder+'/'+input_file)

            appl_list_all = ["Gene3D","PANTHER","Pfam","Coils","SUPERFAMILY","SFLD","Hamap",
                             "ProSiteProfiles","SMART","CDD","PRINTS","PIRSR","ProSitePatterns","AntiFam",
                             "MobiDBLite","PIRSF","FunFam","NCBIfam"]

            for appl in not_exclude:
                if appl not in appl_list_all:
                    raise ValueError('Application not found. Available applications: '+' ,'.join(appl_list_all))

            appl_list = []
            for appl in appl_list_all:
                if appl not in not_exclude:
                    appl_list.append(appl)

            output_file = 'output/interproscan_output_'+bin_index+'.tsv'

            command = "cd "+job_folder+"\n"
            command += "Path=$(pwd)\n"
            command += "bash /home/bubbles/Programs/interproscan-"+version+"/interproscan.sh" # Only in bubbles
            command += " -i $Path/"+input_file
            command += " -f "+output_format
            command += " -o $Path/"+output_file
            command += " -cpu "+str(cpus)
            for n,appl in enumerate(appl_list):
                if n == 0:
                    command += " -exclappl "+appl
                else:
                    command += ","+appl
            command += "\n"

            command += "cd ..\n"

            jobs.append(command)

        print('Remember, Interproscan is only installed in bubbles at the moment')

        return jobs

    def readInterProScanFoldDefinitions(self, job_folder, return_missing=False, verbose=True):
        """
        Reads the output generated by the setUpInterProScan calculation.
        """

        # Check code in input files
        input_codes = set()
        batch_input_codes = {}
        for f in sorted(os.listdir(job_folder+'/input_fasta')):
            batch = f.split('.')[0].split('_')[-1]
            batch_input_codes[batch] = set()
            if not f.endswith('.fasta'):
                continue

            with open(job_folder+'/input_fasta/'+f) as ff:
                for l in ff:
                    if l.startswith('>'):
                        code = l[1:].strip()
                        input_codes.add(code)
                        batch_input_codes[batch].add(code)

        folds = {}
        batch_folds = {}
        for f in sorted(os.listdir(job_folder+'/output')):

            batch = f.split('.')[0].split('_')[-1]
            batch_folds[batch] = set()

            if not f.endswith('.tsv'):
                continue

            with open(job_folder+'/output/'+f) as tsv:
                for l in tsv:
                    code = l.split('\t')[0]
                    fold = l.split('\t')[12]
                    lr = int(l.split('\t')[6])
                    ur = int(l.split('\t')[7])
                    folds.setdefault(code, {})
                    folds[code].setdefault(fold, [])
                    folds[code][fold].append((lr, ur))
                    batch_folds[batch].add(code)

        diff = len(input_codes) - len(folds)
        if diff > 0 and verbose:
            m  = f'There are {diff} missing codes from the output. '
            m += f'Give return_missing=True to return them as a list.'
            print(m)
            for batch in batch_input_codes:
                if batch not in batch_folds:
                    print(f'\tBatch {batch} has no output')
                else:
                    batch_diff = len(batch_input_codes[batch])-len(batch_folds[batch])
                    if batch_diff > 0:
                        print(f'\tThere are {batch_diff} missing codes from the output of batch {batch}.')

        if return_missing:
            missing = []
            for code in input_codes:
                if code not in folds:
                    missing.append(code)

            return missing

        return folds

    def __iter__(self):
        #returning __iter__ object
        self._iter_n = -1
        self._stop_inter = len(self.sequences_names)
        return self

    def __next__(self):
        self._iter_n += 1
        if self._iter_n < self._stop_inter:
            return self.sequences_names[self._iter_n]
        else:
            raise StopIteration

def _structuralClustering(job_folder, models_folder, output_dcd=True, save_as_pdb=False,
                          model_prefix=None, c=0.9, cov_mode=0, evalue=10.0, overwrite=False,
                          stderr=True, stdout=True, verbose=True):
    """
    Perform structural clustering on the PDB files in models_folder using foldseek.
    Clusters are renamed as cluster_01, cluster_02, etc.
    Optionally, each cluster’s structures are loaded and saved as a DCD file.
    Additionally, if save_as_pdb is True, all cluster member PDBs are copied into a single folder
    with filenames of the form: {model_prefix}_{cluster}_{member}.pdb.
    The function caches the output to disk so that subsequent calls will
    simply recover previous results unless overwrite is True. When overwrite is True,
    previous output folders and cache files are deleted.

    Parameters
    ----------
    job_folder : str
        Path where the clustering job will run.
    models_folder : str
        Path to the folder containing the input PDB models.
    output_dcd : bool, optional
        If True, save clustered structures as DCD files (default: True).
    save_as_pdb : bool, optional
        If True, copy all cluster member PDBs into a single folder using the naming format
        {model_prefix}_{cluster}_{member}.pdb (default: False).
    model_prefix : str, optional
        A prefix to use for naming the output PDB files. If None, the basename of models_folder is used.
    c : float, optional
        Fraction of aligned residues for clustering (default: 0.9).
    cov_mode : int, optional
        Coverage mode for clustering (default: 0).
    evalue : float, optional
        E-value threshold for clustering (default: 10.0).
    overwrite : bool, optional
        If True, previous clustering output is deleted and re-run (default: False).

    Returns
    -------
    clusters : dict
        Dictionary where keys are "cluster_XX" and values are dicts with keys "centroid"
        and "members" (a list of member names).
    """

    # Manage stdout and stderr
    if stdout:
        stdout = None
    else:
        stdout = subprocess.DEVNULL

    if stderr:
        stderr = None
    else:
        stderr = subprocess.DEVNULL

    # Convert paths to absolute.
    job_folder = os.path.abspath(job_folder)
    models_folder = os.path.abspath(models_folder)

    if not os.path.exists(job_folder):
        os.mkdir(job_folder)

    # Define cache and output file paths.
    cache_file = os.path.join(job_folder, "clusters.json")
    cluster_output_file = os.path.join(job_folder, "result_cluster.tsv")
    dcd_folder = os.path.join(job_folder, "clustered_dcd")
    pdb_folder = os.path.join(job_folder, "clustered_pdb")  # For saving PDB copies

    # If overwrite is requested, remove previous outputs.
    if overwrite:
        if os.path.exists(cache_file):
            os.remove(cache_file)
        if os.path.exists(cluster_output_file):
            os.remove(cluster_output_file)
        if os.path.exists(dcd_folder):
            shutil.rmtree(dcd_folder)
        if os.path.exists(pdb_folder):
            shutil.rmtree(pdb_folder)

    # If cache exists and not overwriting, load and return.
    if os.path.exists(cache_file) and not overwrite:
        print("Loading cached clusters from", cache_file)
        with open(cache_file, "r") as cf:
            renamed_clusters = json.load(cf)
        return renamed_clusters

    # Create temporary folder for foldseek.
    tmp_folder = os.path.join(job_folder, 'tmp')
    if overwrite and os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    clusters_temp = {}

    # If the foldseek output exists and we're not overwriting, use it.
    if os.path.exists(cluster_output_file) and not overwrite:
        if verbose:
            print("Existing foldseek clustering output found. Reading clusters...")
        with open(cluster_output_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                centroid = parts[0].replace('.pdb', '')
                member = parts[1].replace('.pdb', '')
                clusters_temp.setdefault(centroid, []).append(member)
        print(f"Found {len(clusters_temp)} clusters from foldseek output.")
    else:
        # Build and run the foldseek easy-cluster command.
        command = f"cd {job_folder}\n"
        command += (f"foldseek easy-cluster {models_folder} result tmp "
                    f"--cov-mode {cov_mode} -e {evalue} -c {c}\n")
        command += f"cd {'../'*len(job_folder.split(os.sep))}\n"
        if verbose:
            print("Running foldseek clustering...")
        subprocess.run(command, shell=True, stdout=stdout, stderr=stderr)

        if not os.path.exists(cluster_output_file):
            raise FileNotFoundError(f"Clustering output file not found: {cluster_output_file}")

        with open(cluster_output_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                centroid = parts[0].replace('.pdb', '')
                member = parts[1].replace('.pdb', '')
                clusters_temp.setdefault(centroid, []).append(member)
        if verbose:
            print(f"Clustering complete. Found {len(clusters_temp)} clusters.")

    # Sort clusters by size and rename them as cluster_01, cluster_02, etc.
    clusters_sorted = sorted(clusters_temp.items(), key=lambda x: len(x[1]), reverse=True)
    renamed_clusters = {}
    for i, (centroid, members) in enumerate(clusters_sorted, start=1):
        cluster_name = f"cluster_{i:02d}"
        renamed_clusters[cluster_name] = {"centroid": centroid, "members": members}

    # Optionally, generate a DCD file for each cluster.
    if output_dcd:
        if os.path.exists(dcd_folder):
            shutil.rmtree(dcd_folder)
        os.mkdir(dcd_folder)
        for cluster_name, data in renamed_clusters.items():
            centroid = data["centroid"]
            members = data["members"]
            pdb_names = [centroid] + members
            pdb_files = [os.path.join(models_folder, f"{name}.pdb") for name in pdb_names]

            traj_list = []
            for pdb in pdb_files:
                if os.path.exists(pdb):
                    try:
                        traj = md.load(pdb)
                        traj_list.append(traj)
                    except Exception as e:
                        print(f"Error loading {pdb}: {e}")
                else:
                    print(f"Warning: {pdb} not found.")
            if not traj_list:
                print(f"No valid PDB files found for {cluster_name}. Skipping DCD generation.")
                continue
            try:
                combined_traj = traj_list[0] if len(traj_list) == 1 else md.join(traj_list)
            except Exception as e:
                print(f"Error joining trajectories for {cluster_name}: {e}")
                continue

            dcd_file = os.path.join(dcd_folder, f"{cluster_name}.dcd")
            combined_traj.save_dcd(dcd_file)
            if verbose:
                print(f"Saved {cluster_name} as DCD file: {dcd_file}")

    # Optionally, save the clusters as individual PDBs in one folder.
    if save_as_pdb:
        if os.path.exists(pdb_folder):
            shutil.rmtree(pdb_folder)
        os.mkdir(pdb_folder)
        # Use model_prefix if provided; otherwise derive it from the basename of models_folder.
        prefix = model_prefix if model_prefix is not None else os.path.basename(models_folder)
        for cluster_name, data in renamed_clusters.items():
            for member in [data["centroid"]] + data["members"]:
                source_file = os.path.join(models_folder, f"{member}.pdb")
                if os.path.exists(source_file):
                    target_file = os.path.join(pdb_folder, f"{prefix}_{cluster_name}_{member}.pdb")
                    shutil.copyfile(source_file, target_file)
                    if verbose:
                        print(f"Copied {source_file} to {target_file}")
                else:
                    print(f"Warning: {source_file} not found. Skipping copy.")

    # Clean up temporary folder.
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    # Cache the clusters to disk.
    with open(cache_file, "w") as cf:
        json.dump(renamed_clusters, cf, indent=2)
    if verbose:
        print("Clusters cached to", cache_file)

    return renamed_clusters

def _copyScriptFile(
    output_folder, script_name, no_py=False, subfolder=None, hidden=True, path="prepare_proteins/scripts",
):
    """
    Copy a script file from the prepare_proteins package.

    Parameters
    ==========

    """
    # Get script

    if subfolder != None:
        path = path + "/" + subfolder

    script_file = resource_stream(
        Requirement.parse("prepare_proteins"), path + "/" + script_name
    )
    script_file = io.TextIOWrapper(script_file)

    # Write control script to output folder
    if no_py == True:
        script_name = script_name.replace(".py", "")

    if hidden:
        output_path = output_folder + "/._" + script_name
    else:
        output_path = output_folder + "/" + script_name

    with open(output_path, "w") as sof:
        for l in script_file:
            sof.write(l)
