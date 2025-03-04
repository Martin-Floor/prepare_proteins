from . import alignment
import os
import shutil
import bz2
import pickle
import pandas as pd
import mdtraj as md
import json
import subprocess

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

    def setUpAlphaFold_tunned_mn(self, job_folder, model_preset='monomer_ptm', exclude_finished=True,
                                 remove_extras=False, remove_msas=False, nstruct=1, nrecycles=1,
                                 max_extra_msa=None, keep_compress=False):
        """
        Set up AlphaFold predictions for the loaded sequences. This is a tunned
        version adapted from https://github.com/bjornwallner/alphafoldv2.2.0
        """

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
                for f in os.listdir(job_folder+'/output_models/'+model):
                    if f == 'ranked_0.pdb' or f == 'ranked__0.pdb.bz2':
                        excluded.append(model)

        jobs = []
        for model in self.sequences:
            if exclude_finished and model in excluded:
                continue
            sequence = {}
            sequence[model] = self.sequences[model]
            alignment.writeFastaFile(sequence, job_folder+'/input_sequences/'+model+'.fasta')
            command = 'cd '+job_folder+'\n'
            command += 'Path=$(pwd)\n'
            command += 'singularity run -B $ALPHAFOLD_DATA_PATH:/data -B /gpfs/projects/bsc72/alphafold_tunned/alphafoldv2.2.0:/app/alphafold --pwd /app/alphafold --nv $ALPHAFOLD_CONTAINER --data_dir=/data --uniref90_database_path=/gpfs/projects/shared/public/AlphaFold/uniref90/uniref90.fasta --mgnify_database_path=/gpfs/projects/shared/public/AlphaFold/mgnify/mgy_clusters_2018_12.fa --uniclust30_database_path=/gpfs/projects/shared/public/AlphaFold/uniclust30/uniclust30_2018_08/uniclust30_2018_08 --bfd_database_path=/gpfs/projects/shared/public/AlphaFold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt --pdb70_database_path=/gpfs/projects/shared/public/AlphaFold/pdb70/pdb70 --template_mmcif_dir=/data/pdb_mmcif/mmcif_files'
            command += f' --nstruct={nstruct}'
            command += f' --max_recycles={nrecycles}'
            if max_extra_msa is not None:
                command += f' --max_extra_msa={max_extra_msa}'
            command += ' --fasta_paths $Path/input_sequences/'+model+'.fasta'
            command += ' --output_dir=$Path/output_models'
            command += ' --model_preset='+model_preset
            command += ' --max_template_date=2022-01-01'
            command += ' --random_seed 1 --obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat "$@"\n'

            if keep_compress==False:
                command += 'bzip2 -d *.pdb.bz2\n'

            if remove_extras:
                command += f'rm -r $Path/output_models/{model}/msas\n'
                command += f'rm -r $Path/output_models/{model}/*.pkl\n'

            if remove_msas:
                command += f'rm -r $Path/output_models/{model}/msas\n'

            command += 'cd ..\n'

            jobs.append(command)

        return jobs

    def setUpAlphaFold_tunned_mt(self, job_folder, model_preset='monomer_ptm', exclude_finished=True,
                       remove_extras=False, remove_msas=False,nstruct=1,nrecycles=1,max_extra_msa=None,keep_compress=False):
        """
        Set up AlphaFold predictions for the loaded sequneces. This is a tunned version adapted from https://github.com/bjornwallner/alphafoldv2.2.0

        """

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
                for f in os.listdir(job_folder+'/output_models/'+model):
                    if f == 'ranked_0.pdb':
                        excluded.append(model)

        jobs = []
        for model in self.sequences:
            if exclude_finished and model in excluded:
                continue
            sequence = {}
            sequence[model] = self.sequences[model]
            alignment.writeFastaFile(sequence, job_folder+'/input_sequences/'+model+'.fasta')
            command = 'cd '+job_folder+'\n'
            command += 'Path=$(pwd)\n'
            command += 'singularity run -B $ALPHAFOLD_DATA_PATH:/data -B /opt/cuda/10.1,.:/etc,$TMPDIR:/tmp -B /gpfs/projects/bsc72/alphafold_tunned/alphafoldv2.2.0:/app/alphafold --pwd /app/alphafold --nv $ALPHAFOLD_CONTAINER --data_dir=/data --uniref90_database_path=/gpfs/projects/shared/public/AlphaFold/uniref90/uniref90.fasta --mgnify_database_path=/gpfs/projects/shared/public/AlphaFold/mgnify/mgy_clusters_2018_12.fa --uniclust30_database_path=/gpfs/projects/shared/public/AlphaFold/uniclust30/uniclust30_2018_08/uniclust30_2018_08 --bfd_database_path=/gpfs/projects/shared/public/AlphaFold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt --pdb70_database_path=/gpfs/projects/shared/public/AlphaFold/pdb70/pdb70 --template_mmcif_dir=/data/pdb_mmcif/mmcif_files'
            command += f' --nstruct={nstruct}'
            command += f' --max_recycles={nrecycles}'
            if max_extra_msa is not None:
                command += f' --max_extra_msa={max_extra_msa}'
            command += ' --fasta_paths $Path/input_sequences/'+model+'.fasta'
            command += ' --output_dir=$Path/output_models'
            command += ' --model_preset='+model_preset
            command += ' --max_template_date=2022-01-01'
            command += ' --random_seed 1 --obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat "$@"\n'

            if keep_compress==False:
                command += 'bzip2 -d *.pdb.bz2\n'

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

    def setUpBioEmu(self, job_folder, num_samples=10000, batch_size_100=20, gpu_local=False):

        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        jobs = []
        for model in self.sequences:

            model_folder = job_folder+'/'+model
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)

            command = ''
            if gpu_local:
                command += 'CUDA_VISIBLE_DEVICES=GPUID '
            command += 'python -m bioemu.sample '
            command += f'--sequence {self.sequences[model]} '
            command += f'--num_samples {num_samples} '
            command += f'--batch_size_100 {batch_size_100} '
            command += f'--output_dir {model_folder}\n'

            jobs.append(command)

        return jobs

    def clusterBioEmuSamples(self, job_folder, bioemu_folder, models=None, stderr=True, stdout=True,
                             output_dcd=False, output_pdb=False, c=0.9, cov_mode=0, verbose=True,
                             evalue=10.0, overwrite=False, remove_input_pdb=True):
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

            # Load the topology and trajectory; superpose the trajectory.
            traj_top = md.load(top_file)
            traj = md.load(traj_file, top=top_file)
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
