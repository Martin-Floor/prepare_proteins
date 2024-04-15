from . import alignment
import os
import shutil
import bz2
import pickle
import pandas as pd

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

    def copyModelsFromAlphaFoldCalculation(self, af_folder, output_folder, prefix='', return_missing=False):
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
                elif f.startswith('ranked__0'):
                    models_paths[d] = mdir+'/'+f

        # Create output folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if return_missing:
            missing = []
        for m in self:
            if m in models_paths:
                if models_paths[m].endswith('.pdb'):
                    shutil.copyfile(models_paths[m], output_folder+'/'+prefix+m+'.pdb')
                elif models_paths[m].endswith('.bz2'):
                    file = bz2.BZ2File(models_paths[m], 'rb')
                    pdbfile = open(output_folder+'/'+prefix+m+'.pdb', 'wb')
                    shutil.copyfileobj(file, pdbfile)
            else:
                if return_missing:
                    missing.append(m)
                print('Alphafold model for sequence %s was not found in folder %s' % (m, af_folder))

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

    def setUpInterProScan(self, job_folder, not_exclude=['Gene3D'], output_format='tsv',
                          cpus=40, version="5.67-99.0", max_bin_size=10000):
        """
        Set up InterProScan analysis to search for domains in a set of proteins

        not_exclude: list
            refers to programs that will used to find the domains, and that will retrieve results. This has an
            impact on the time.

        If an update is needed, download in bubbles the new version (replace XX-XX with the version number):
            wget -O interproscan-5.XX-XX.0-64-bit.tar.gz http://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.XX-XX.0/interproscan-5.67-99.0-64-bit.tar.gz
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
        n_bins = len(self.sequences) // 10000
        if len(self.sequences) % 10000 > 0:
            n_bins += 1
        zf = len(str(n_bins))

        # Get all sequences names
        all_sequences = list(self.sequences.keys())

        # Create commands for each sequence bin
        jobs = []
        for bin in range(1, n_bins+1):

            bin_index = str(bin).zfill(zf)

            input_file = 'input_fasta/input_'+bin_index+'.fasta'

            bin_sequences = {s:self.sequences[s] for s in all_sequences[bin*10000:(bin+1)*10000]}

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
            command += "bash /home/bubbles/Programs/interproscan/interproscan-"+version+"/interproscan.sh" # Only in bubbles
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

    def readInterProScanFoldDefinitions(job_folder):
        """
        Reads the output generated by the setUpInterProScan calculation.
        """

        folds = {}
        for f in sorted(os.listdir(job_folder+'/output')):

            if not f.endswith('.tsv'):
                conitnue

            with open(job_folder+'/output/'+f) as tsv:
                for l in tsv:
                    code = l.split('\t')[0]
                    fold = l.split('\t')[12]
                    lr = int(l.split('\t')[6])
                    ur = int(l.split('\t')[7])
                    folds.setdefault(code, {})
                    folds[code][fold] = (lr, ur)

        return folds
