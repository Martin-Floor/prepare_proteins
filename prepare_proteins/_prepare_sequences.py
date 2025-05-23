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
from tqdm import tqdm
import ipywidgets as widgets
from ipywidgets import interact
from collections import defaultdict

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

            depth = len(os.path.normpath(job_folder).split(os.sep))
            command += 'cd '+'../'*depth+'\n'
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
                    verbose=True, models=None, skip_finished=False, return_finished=False,
                    filter_samples=True, alphafold_folder=None,
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

        local_af = False
        external_af = False
        af_jobs = None
        if alphafold_folder and os.path.exists(alphafold_folder):

            if not os.path.exists(alphafold_folder+'/output_models'):
                raise ValueError(f'AF folder exists {alphafold_folder} but has an incorrect format!')

            msa_file = {}
            for model in os.listdir(alphafold_folder+'/output_models'):
                model_folder = os.path.join(alphafold_folder+'/output_models', model)
                if not os.path.exists(model_folder):
                    raise ValueError(f'AF folder exists {alphafold_folder} but model {model_folder} has no output!')
                for f in os.listdir(model_folder+'/msas/'):
                    if f.endswith('.a3m'):
                        msa_file[model] = model_folder+'/msas/'+f
                if model not in msa_file:
                    raise ValueError(f'The given AF folder {alphafold_folder} exists but does not contain an MSA for model {model}.\
                    Please consider running the AF calculation first and get the MSAs or use a local AF folder inside bioemu\
                    Just give a folder that do no exist and the function will do the rest.')
            external_af = True

        elif alphafold_folder and os.path.exists(job_folder+'/'+alphafold_folder):
            local_af = True
            with_msa = []
            af_jobs = {}
            for model in os.listdir(job_folder+'/'+alphafold_folder+'/output_models'):
                model_folder = os.path.join(job_folder+'/'+alphafold_folder+'/output_models', model)
                if not os.path.exsits(model_folder):
                    af_jobs[model] = self.setUpAlphaFold(job_folder+'/'+alphafold_folder, only_model=model)
                else:
                    for f in os.listdir(model_folder+'/msas/'):
                        if f.endswith('.a3m'):
                            with_msa.append(model)
                    if model not in with_msa:
                        af_jobs[model] = self.setUpAlphaFold(job_folder+'/'+alphafold_folder, only_model=model)

            if len(list(os.listdir(job_folder+'/'+alphafold_folder+'/output_models'))) == 0:
                af_jobs = self.setUpAlphaFold(job_folder+'/'+alphafold_folder, exclude_finished=False)

        elif alphafold_folder and not os.path.exists(job_folder+'/'+alphafold_folder):
            local_af = True
            af_jobs = self.setUpAlphaFold(job_folder+'/'+alphafold_folder, exclude_finished=False)

        jobs = []
        finished = []
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
                        finished.append(model)
                        continue

            cache_dir = os.path.join(model_folder, 'cache')
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)

            if bioemu_env:

                cached_files = [f for f in os.listdir(cache_dir)]
                fasta_cached_file = [f for f in cached_files if f.endswith('.fasta')]
                npy_cached_files = [f for f in cached_files if f.endswith('.npy')]
                npz_cached_files = [f for f in cached_files if f.endswith('.npz')]

                if len(fasta_cached_file) == 1 and len(npy_cached_files) == 2 and len(npz_cached_files) == 3:
                    if verbose:
                        print(f'Input files for model {model} were found.')
                else:
                    command = f"""
                    source {conda_sh}
                    conda activate {bioemu_env}
                    python -m bioemu.sample --sequence {self.sequences[model]} --num_samples 1 --batch_size_100 {batch_size_100} --cache_embeds_dir {cache_dir} --cache_so3_dir {cache_dir} --output_dir {model_folder}
                    conda deactivate
                    """
                    if verbose:
                        print(f"Setting input files for model {model}")
                    result = subprocess.run(["bash", "-i", "-c", command], capture_output=True, text=True)

            if external_af:
                msa_folder = os.path.join(model_folder, 'msa')
                os.makedirs(msa_folder, exist_ok=True)
                shutil.copyfile(msa_file[model], msa_folder+'/'+model+'.a3m')

            command = 'RUN_SAMPLES='+str(num_samples)+'\n'
            if filter_samples:
                command += 'while true; do\n'
            #command += 'FILE_COUNT=$(find "'+job_folder+'/'+model+'/batch*'+'" -type f | wc -l)\n'
            if gpu_local:
                command += 'CUDA_VISIBLE_DEVICES=GPUID '
            command += 'python -m bioemu.sample '
            if external_af:
                msaf = msa_folder+'/'+model+'.a3m'
                command += f'--sequence {msaf} '
            elif local_af:
                msaf = job_folder+'/'+alphafold_folder+'/output_models/'+model+'/msas/bfd_uniref_hits.a3m '
                command += f'--sequence {msaf} '
            else:
                command += f'--sequence {self.sequences[model]} '
            command += f'--num_samples $RUN_SAMPLES '
            command += f'--batch_size_100 {batch_size_100} '
            command += f'--cache_embeds_dir {cache_dir} '
            command += f'--cache_so3_dir {cache_dir} '
            if not filter_samples:
                command += f'--filter_samples 0 '
            command += f'--output_dir {model_folder}\n'
            if filter_samples:
                command += 'NUM_SAMPLES=$(python -c "import mdtraj as md; traj = md.load_xtc(\''+job_folder+'/'+model+'/samples.xtc\', top=\''+job_folder+'/'+model+'/topology.pdb\'); print(traj.n_frames)")\n'
                command += 'if [ "$NUM_SAMPLES" -ge '+str(num_samples)+' ]; then\n'
                command += 'echo "All samples computed. Exiting."\n'
                command += 'break \n'
                command += 'fi\n'
                command += 'RUN_SAMPLES=$(($RUN_SAMPLES+'+str(num_samples)+'-$NUM_SAMPLES))\n'
                command += 'done \n'

            if isinstance(af_jobs, dict):
                command = af_jobs[model]+command
            jobs.append(command)

        # Combine jobs if AF folder was not found
        if isinstance(af_jobs, list):
            jobs = [a+b for a,b in zip(af_jobs,jobs)]

        if return_finished:
            return finished

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

    def computeBioEmuRMSF(self, bioemu_folder, ref_pdb=None, plot=False, ylim=None, plot_legend=True):
        """
        Computes RMSF values for all models in the specified folder and optionally plots RMSF by residue.

        Parameters:
        - bioemu_folder (str): Path to the folder containing model subdirectories with trajectory and topology files.
        - ref_pdb (dict): Dictionary with paths to the reference PDB structures for RMSF calculation for each model.
        - plot (bool, optional): Whether to generate a plot of RMSF vs. residue. Default is False.

        Returns:
        - rmsf (dict): Dictionary containing RMSF arrays for each model.
          Each array holds the per-residue RMSF (in nm) for the selected backbone (Cα) atoms.
        """
        # Load reference structure and select backbone Cα atoms
        if isinstance(ref_pdb, str):
            unique_ref = ref_pdb
            ref_pdb = {}
            for model in os.listdir(bioemu_folder):
                ref_pdb[model] = unique_ref

        if not ref_pdb:
            print('No reference PDBs given. Computing RMSF relative to the average positions')

        # Dictionary to store RMSF values for each model
        rmsf = {}

        # Iterate through each model folder
        for model in os.listdir(bioemu_folder):

            # Load reference structure
            if ref_pdb and model in ref_pdb:
                ref = md.load(ref_pdb[model])
                ref_bb_atoms = ref.topology.select('name CA')

            traj_file = f'{bioemu_folder}/{model}/samples.xtc'
            top_file = f'{bioemu_folder}/{model}/topology.pdb'

            if not os.path.exists(traj_file):
                continue

            traj = md.load(traj_file, top=top_file)
            traj_bb_atoms = traj.topology.select('name CA')

            if not ref_pdb:
                ref = md.load(top_file)
                ref.xyz = np.mean(ref.xyz, axis=0) # Set reference to the average positions

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
            if plot_legend:
                plt.legend()
            plt.tight_layout()

        return rmsf

    def computeBioEmuRMSD(self, bioemu_folder, ref_pdb, residues=None, plot=False):
        """
        Computes RMSD values for all models in the specified folder and optionally plots a violin plot.

        Parameters:
        - bioemu_folder (str): Path to the folder containing model subdirectories with trajectory and topology files.
        - ref_pdb (dict): Dictionary with paths to the reference PDB structure for RMSD calculation for each model.
        - plot (bool, optional): Whether to generate a violin plot. Default is True.

        Returns:
        - rmsd (dict): Dictionary containing RMSD arrays for each model.
        """

        if isinstance(ref_pdb, str):
            unique_ref = ref_pdb
            ref_pdb = {}
            for model in os.listdir(bioemu_folder):
                ref_pdb[model] = unique_ref

        # Dictionary to store RMSD values
        rmsd = {}

        # Iterate through each model folder
        for model in os.listdir(bioemu_folder):

            # Load reference structure
            ref = md.load(ref_pdb[model])

            ref_bb_atoms = []
            for residue in ref.topology.residues:
                if residues and residue.resSeq not in residues:
                    continue
                for atom in residue.atoms:
                    if atom.name == 'CA':
                        ref_bb_atoms.append(atom.index)
            ref_bb_atoms = np.array(ref_bb_atoms)

            traj_file = f'{bioemu_folder}/{model}/samples.xtc'
            top_file = f'{bioemu_folder}/{model}/topology.pdb'

            if not os.path.exists(traj_file):
                continue

            traj = md.load(traj_file, top=top_file)

            traj_bb_atoms = []
            for residue in traj.topology.residues:
                if residues and residue.resSeq not in residues:
                    continue
                for atom in residue.atoms:
                    if atom.name == 'CA':
                        traj_bb_atoms.append(atom.index)
            traj_bb_atoms = np.array(traj_bb_atoms)

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

    def computeNativeContacts(self, job_folder, bioemu_folder, native_models_folder):
        """
        Compute the distances between native contacts across frames for each model in the dataset.

        For each model:
        - Removes hydrogens and OXT atoms from the native PDB file.
        - Uses SMOG2 to define native contacts (C-alpha based).
        - Computes distances between native contacts over the trajectory.
        - Returns a DataFrame for each model, where:
            - Rows = frames
            - Columns = native contacts (formatted as "resA-resB")
            - Values = distances in nm

        Parameters:
        - job_folder (str): Path where temporary SMOG files will be stored.
        - bioemu_folder (str): Path to BioEmu output folders, one per model (should include topology and trajectory).
        - native_models_folder (str, dict): Path to PDB folder containg the files of native models, one per model. Alternatively,
                                            a dictionary containing the paths to each native PDB file.

        Returns:
        - df_distances (dict): Dictionary with model names as keys and DataFrames as values.
                               Each DataFrame contains native contact distances across frames.
        """

        def remove_hydrogens_and_oxt(pdb_path, output_path):
            """Removes all hydrogen atoms and OXT atoms from a PDB file."""
            with open(pdb_path) as f_in, open(output_path, 'w') as f_out:
                for line in f_in:
                    if line.startswith("TER") or line.startswith("END"):
                        continue
                    if not line.startswith("ATOM"):
                        f_out.write(line)
                        continue

                    atom_name = line[12:16].strip()
                    element = line[76:78].strip()
                    if element == 'H' or atom_name.startswith('H') or atom_name == 'OXT':
                        continue

                    f_out.write(line)
                f_out.write('END\n')

        def readNC(nc_file):
            """Reads a .contacts.CG file and returns a list of (res1, res2) tuples."""
            contacts = []
            for l in open(nc_file):
                contacts.append((int(l.split()[1]), int(l.split()[3])))
            return contacts

        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if isinstance(native_models_folder, str) and os.path.isdir(native_models_folder):
            native_models = {
                m.replace('.pdb', ''): os.path.join(native_models_folder, m)
                for m in os.listdir(native_models_folder) if m.endswith('.pdb')
            }
        elif isinstance(native_models_folder, dict):
            native_models = native_models_folder
        else:
            raise ValueError('native_models_folder should  be a existing path or a dictionary!')

        df_distances = {}

        models = list(self)  # Ensures tqdm knows the total number of items
        for model in tqdm(models, desc="Computing native contacts", ncols=100):

            if model not in native_models:
                raise ValueError(f'Model "{model}" not found in native_models_folder: {native_models_folder}')

            model_folder = os.path.join(job_folder, model)
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)

            contacts_file = os.path.join(model_folder, f'{model}.contacts.CG')
            contacts_pdb = os.path.join(model_folder, f'{model}.pdb')

            # Generate native contacts with SMOG2 if not already done
            if not os.path.exists(contacts_file):
                filtered_pdb = os.path.join(model_folder, f'{model}.pdb')
                remove_hydrogens_and_oxt(native_models[model], filtered_pdb)
                command = f'cd {model_folder} && smog2 -i {model}.pdb -s {model} -CA'
                os.system(command)

            native_contacts = readNC(contacts_file)

            # Load topology and Cα atom indices
            top_file = os.path.join(bioemu_folder, model, 'topology.pdb')
            top_traj = md.load(top_file)
            ca_atoms = [a.index for a in top_traj.topology.atoms if a.name == 'CA']
            native_pairs = [(ca_atoms[c[0]-1], ca_atoms[c[1]-1]) for c in native_contacts]

            # Load trajectory
            traj_file = os.path.join(bioemu_folder, model, 'samples.xtc')
            traj = md.load(traj_file, top=top_file)

            # Compute distances from trajectory
            D = md.compute_distances(traj, native_pairs)

            # Compute Frame 0 (native contact distances) from the filtered native structure
            native_traj = md.load(contacts_pdb)
            ca_atoms = [a.index for a in native_traj.topology.atoms if a.name == 'CA']
            native_pairs = [(ca_atoms[c[0]-1], ca_atoms[c[1]-1]) for c in native_contacts]
            native_distances = md.compute_distances(native_traj, native_pairs)

            # Combine native distances with trajectory distances
            D_all = np.vstack([native_distances, D])  # shape: (n_frames + 1, n_contacts)

            # Create labels like "resA-resB"
            contact_labels = [
                f'{native_contacts[i][0]}-{native_contacts[i][1]}'
                for i in range(len(native_contacts))
            ]

            # Build DataFrame with Frame 0 included
            df = pd.DataFrame(D_all, columns=contact_labels)
            df['Frame'] = range(D_all.shape[0])  # Frame 0 = native
            df = df.set_index('Frame')
            df_distances[model] = df

        return df_distances

    def computeFractionOfNativeContacts(self, df_distances, method="hard",
                                        inflation=1.2, rel_tolerance=0.2):
        """
        Compute the fraction of native contacts (Q) per frame and per model, using either a
        hard cutoff or a relative tolerance.

        This function accepts a dictionary of DataFrames (one per model), as returned by
        computeNativeContacts(). Each DataFrame must have:
          - A row at index=0 containing the native contact distances.
          - Additional rows (index=1..N) for simulation frames.

        The fraction of native contacts (Q) is computed for each frame by determining
        how many contacts are "formed" (according to the chosen method) and then dividing
        by the total number of contacts.

        Supported methods:
          - "hard":
              Each contact is considered formed if its distance in a given frame is below
              the native distance (index=0) multiplied by an inflation factor.

              Q(frame) = (# of contacts with distance < native_distance * inflation) / (# contacts)

          - "relative":
              Each contact is formed if its distance does not exceed the native distance by more
              than a relative tolerance.

              Q(frame) = (# of contacts satisfying ((distance / native_distance) - 1) < rel_tolerance) / (# contacts)

        Parameters
        ----------
        df_distances : dict
            Dictionary with model names as keys and DataFrames as values.
            Each DataFrame has distances per contact (columns) across frames (rows).
            Row index 0 must represent the native (equilibrium) distances.

        method : str, optional
            Choice of method to compute Q. Must be "hard" or "relative".
            Default is "hard".

        inflation : float, optional
            Factor by which the native distance is multiplied when using the "hard" method.
            Default is 1.2.

        rel_tolerance : float, optional
            Relative tolerance for the "relative" method.
            Default is 0.2 (i.e., 20% deviation allowed).

        Returns
        -------
        df_q : pd.DataFrame
            DataFrame with a MultiIndex [Model, Frame] and a single column "Q".
            The "Frame" index excludes row 0 (the native reference).
            "Q" is the fraction of native contacts formed at each frame.
        """
        records = []

        # Iterate over each model and its DataFrame of distances
        for model, df in df_distances.items():
            # 1) Native distances (row 0)
            native = df.loc[0].values.astype(float)  # shape: (n_contacts,)

            # 2) Simulation frames (all rows except 0)
            sim_df = df.drop(index=0)
            frames = sim_df.values.astype(float)  # shape: (n_frames, n_contacts)
            n_contacts = native.shape[0]

            # 3) Compute Q according to the chosen method
            if method == "hard":
                # Hard cutoff = native * inflation
                cutoff = native[None, :] * inflation
                formed = frames < cutoff  # boolean array (n_frames, n_contacts)
                q_vals = formed.sum(axis=1) / n_contacts

            elif method == "relative":
                # Relative tolerance around native distance
                formed = ((frames / native[None, :]) - 1) < rel_tolerance
                q_vals = formed.sum(axis=1) / n_contacts

            else:
                raise ValueError(f"Unsupported method: {method}")

            # 4) Build temporary DataFrame with Q values for each frame
            tmp_df = pd.DataFrame({
                "Model": model,
                "Frame": sim_df.index,  # original frame numbers
                "Q": q_vals
            })
            records.append(tmp_df)

        # Combine results for all models into a single DataFrame with a MultiIndex
        df_q = pd.concat(records, ignore_index=True).set_index(['Model', 'Frame'])
        return df_q

    def plotNativeContactsDistribution(self, q_df):
        """
        Create an interactive KDE plot of the distribution of native contact fraction (Q)
        for a selected model.

        This function uses ipywidgets to create a dropdown menu of models based on the input
        Q dataframe. When a model is selected, it displays a Kernel Density Estimate (KDE)
        plot of Q values for that model.

        Parameters:
          - q_df (pd.DataFrame): DataFrame containing the fraction of native contacts for each frame
                                 and for each model. It is assumed that the index contains the
                                 "Model" level.
        """
        # Reset index to easily filter by the "Model" column.
        q_reset = q_df.reset_index()
        # Get the unique models present in the Q dataframe.
        models = q_reset["Model"].unique()

        def plot_model_q(model):
            """
            Inner function: filters the Q dataframe for the selected model and plots its KDE.
            """
            df_model = q_reset[q_reset['Model'] == model]
            plt.figure(figsize=(8, 6))
            sns.kdeplot(df_model['Q'], shade=True, color="skyblue", bw_adjust=1)
            plt.title(f"KDE of Q for model: {model}", fontsize=16)
            plt.xlabel("Fraction of Native Contacts (Q)", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.tight_layout()
            plt.show()

        # Create a dropdown widget and link it to the plotting function.
        model_dropdown = widgets.Dropdown(options=sorted(models), description='Select Model:')
        interact(plot_model_q, model=model_dropdown)

    def computeFoldingFreeEnergy(self, df_q, q_threshold=0.8, temperature=300.0):
        """
        Compute the free energy of folding (ΔG_f) for each model from Q distributions.

        A frame is considered "folded" if Q ≥ q_threshold, and "unfolded" otherwise.

        ΔG is calculated using:
            ΔG = -RT * ln(P_folded / P_unfolded)

        Parameters:
        ----------
        df_q : pd.DataFrame
            DataFrame with MultiIndex ['Model', 'Frame'] and a 'Q' column.

        q_threshold : float
            Threshold to distinguish folded vs. unfolded (default: 0.5)

        temperature : float
            Temperature in Kelvin (default: 300.0)

        Returns:
        --------
        df_dG : pd.DataFrame
            DataFrame with one row per model and columns:
            ['Model', 'ΔG (kcal/mol)', 'P_folded', 'P_unfolded']
        """
        R = 0.001987  # kcal/mol·K

        data = []
        for model in df_q.index.get_level_values('Model').unique():
            q_vals = df_q.loc[model, 'Q'].values
            total = len(q_vals)
            n_folded = np.sum(q_vals >= q_threshold)
            n_unfolded = total - n_folded

            if n_folded == 0 or n_unfolded == 0:
                dG = np.nan  # cannot compute log(0)
            else:
                P_folded = n_folded / total
                P_unfolded = n_unfolded / total
                dG = -R * temperature * np.log(P_folded / P_unfolded)

                data.append({
                    'Model': model,
                    'ΔG_f (kcal/mol)': dG,
                    'P_folded': P_folded,
                    'P_unfolded': P_unfolded
                })

        df_dG = pd.DataFrame(data).set_index('Model')
        return df_dG

    def setUpInterProScan(self, job_folder, not_exclude=['Gene3D'], output_format='tsv',
                          cpus=40, version="5.74-105.0", max_bin_size=10000):
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

    def plotInterProScanFoldDistributions(self, ips, top_n=10):
        """
        Plots the top N most frequent folds from an InterProScan result dictionary,
        excluding the fold named '-'.

        Parameters:
        -----------
        ips : dict
            Dictionary where keys are model identifiers and values are lists of fold names.
        top_n : int, optional
            Number of top folds to display. Default is 10.
        """
        # Dictionary to store fold counts
        fold_counts = defaultdict(int)

        # Count occurrences of each fold
        for model in ips:
            for fold in ips[model]:
                fold_counts[fold] += 1

        # Exclude the fold named '-'
        filtered_fold_counts = {fold: count for fold, count in fold_counts.items() if fold != '-'}

        # Sort folds by frequency and select the top N
        sorted_folds = sorted(filtered_fold_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Extract fold names and counts
        fold_names, counts = zip(*sorted_folds) if sorted_folds else ([], [])

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.bar(fold_names, counts)
        plt.xlabel("Fold")
        plt.ylabel("Frequency")
        plt.title(f"Top {top_n} Most Frequent Folds in InterProScan Search (Excluding '-')")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plotInterProScanSingleSegmentCountsVsMaxGap(self, ips, fold, vertical_line=None, max_gap_cap=100):
        """
        Analyzes the effect of max_gap on fold stitching and plots the number
        of models resulting in a single stitched segment.

        Parameters:
        -----------
        ips : dict
            InterProScan results. {model_id: {fold_name: [(start, end), ...]}}.
        fold : str
            Fold name to analyze.
        max_gap_cap : int
            Optional cap on max_gap range (default 100).
        """

        def merge_intervals(intervals, max_gap):
            if not intervals:
                return []
            intervals = sorted(intervals, key=lambda x: x[0])
            merged = [intervals[0]]
            for start, end in intervals[1:]:
                last_start, last_end = merged[-1]
                if start - last_end <= max_gap:
                    merged[-1] = (last_start, max(last_end, end))
                else:
                    merged.append((start, end))
            return merged

        # Precompute relevant models and gaps
        model_intervals = []
        max_gap_found = 0

        for model, annotations in ips.items():
            if fold in annotations:
                intervals = sorted(annotations[fold], key=lambda x: x[0])
                model_intervals.append(intervals)
                for i in range(1, len(intervals)):
                    gap = intervals[i][0] - intervals[i-1][1]
                    if gap > 0:
                        max_gap_found = max(max_gap_found, gap)

        max_gap = min(max_gap_found, max_gap_cap)

        # Count single-segment models for each gap
        gap_list, count_list = [], []

        for gap in range(max_gap + 1):
            count = sum(1 for intervals in model_intervals if len(merge_intervals(intervals, gap)) == 1)
            gap_list.append(gap)
            count_list.append(count)

        # Plot
        plt.figure(figsize=(8, 5))

        if vertical_line:
            plt.axvline(vertical_line, c='k', ls='--', lw=0.5)

        plt.plot(gap_list, count_list, marker='o')
        plt.xlabel('max_gap (residues)')
        plt.ylabel('Number of Single-Segment Models')
        plt.title(f"Effect of max_gap on Fold Stitching: {fold}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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
