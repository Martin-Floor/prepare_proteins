from . import alignment
import os

class sequenceModels:

    def __init__(self, models_fasta):

        self.sequences = alignment.readFastaFile(models_fasta, replace_slash=True)

    def setUpAlphaFold(self, job_folder, model_preset='monomer_ptm', exclude_finished=True,
                       remove_extras=True):
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
                command += 'rm -r $Path/output_models/msas\n'
                command += 'rm -r $Path/output_models/*.pkl\n'

            command += 'cd ..\n'

            jobs.append(command)

        return jobs
