from . import alignment
import os

class sequenceModels:

    def __init__(self, models_fasta):

        self.sequences = alignment.readFastaFile(models_fasta, replace_slash=True)

    def setUpAlphaFold(self, job_folder, model_preset='monomer_ptm'):
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

        jobs = []
        for model in self.sequences:
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
            command += 'cd ..\n'

            jobs.append(command)

        return jobs
