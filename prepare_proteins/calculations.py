import bsc_calculations
import os

def setUpPELEForMarenostrum(jobs, partition='bsc_ls', cpus=96):
    """
    Creates submission scripts for Marenostrum for each PELE job inside the jobs variable.

    Parameters
    ==========
    jobs : list
        Commands for run PELE. This is the output of the setUpPELECalculation() function.
    """
    if not os.path.exists('pele_slurm_scripts'):
        os.mkdir('pele_slurm_scripts')

    zfill = len(str(len(jobs)))
    with open('pele_slurm.sh' , 'w') as ps:
        for i,job in enumerate(jobs):
            job_name = str(i+1).zfill(zfill)+'_'+job.split('\n')[0].split('/')[-1]
            bsc_calculations.marenostrum.singleJob(job, cpus=cpus, partition=partition, program='pele',
                                                   job_name=job_name, script_name='pele_slurm_scripts/'+job_name+'.sh')
            ps.write('sbatch pele_slurm_scripts/'+job_name+'.sh\n')
