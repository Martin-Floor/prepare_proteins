from . import alignment
import os
import shutil

class sequenceModels:

    def __init__(self, models_fasta):

        self.sequences = alignment.readFastaFile(models_fasta, replace_slash=True)
        self.sequences_names = list(self.sequences.keys())

    def setUpAlphaFold(self, job_folder, model_preset='monomer_ptm', exclude_finished=True,
                       remove_extras=False, remove_msas=False):
        """
        Set up AlphaFold predictions for the loaded sequneces
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
            command += 'bsc_alphafold --fasta_paths $Path/input_sequences/'+model+'.fasta'
            command += ' --output_dir=$Path/output_models'
            command += ' --model_preset='+model_preset
            command += ' --max_template_date=2022-01-01'
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
            command += 'singularity run -B $ALPHAFOLD_DATA_PATH:/data -B /gpfs/projects/bsc72/alphafold_tunned/alphafoldv2.2.0:/app/alphafold --pwd /app/alphafold --nv $ALPHAFOLD_CONTAINER --data_dir=/data --uniref90_database_path=/gpfs/projects/shared/public/AlphaFold/uniref90/uniref90.fasta --mgnify_database_path=/gpfs/projects/shared/public/AlphaFold/mgnify/mgy_clusters_2018_12.fa --uniclust30_database_path=/gpfs/projects/shared/public/AlphaFold/uniclust30/uniclust30_2018_08/uniclust30_2018_08 --bfd_database_path=/gpfs/projects/shared/public/AlphaFold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt --pdb70_database_path=/gpfs/projects/shared/public/AlphaFold/pdb70/pdb70 --template_mmcif_dir=/data/pdb_mmcif/mmcif_files '
            command += f' --nstruct={nstruct}'
            command += f' --max_recycles={nrecycles}'
            if max_extra_msa =! None:
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
    

    def copyModelsFromAlphaFoldCalculation(self, af_folder, output_folder):
        """
        Copy models from an AlphaFold calculation to an specfied output folder.

        Parameters
        ==========
        af_folder : str
            Path to the Alpha fold folder calculation
        output_folder : str
            Path to the output folder where to store the models
        """

        # Get paths to models in alphafold folder
        models_paths = {}
        for d in os.listdir(af_folder+'/output_models'):
            mdir = af_folder+'/output_models/'+d
            for f in os.listdir(mdir):
                if f.startswith('relaxed_model_1_ptm'):
                    models_paths[d] = mdir+'/'+f

        # Create output folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for m in self:
            if m in models_paths:
                shutil.copyfile(models_paths[m], output_folder+'/'+m+'.pdb')
            else:
                print('Alphafold model for sequence %s was not found in folder %s' % (m, af_folder))

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
