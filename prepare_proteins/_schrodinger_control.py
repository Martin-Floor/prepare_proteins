import argparse
import time
import os

# ## Define input variables
parser = argparse.ArgumentParser()

parser.add_argument('log_file', help='Path to the logfile generated.')
parser.add_argument('--job_type', help='Type of calculation launched by Schrodinger.')

args=parser.parse_args()

log_file = args.log_file
job_type = args.job_type

implemented_calculations = [
'prepwizard',
'grid',
'docking'
]

if job_type not in implemented_calculations:
    raise ValueError(job_type+' not implemented in the Schrodinger control script.')

# Wait for the log file to be generated
while not os.path.exists(log_file):
    time.sleep(1)

print('Checking log file: '+log_file, end=' ')
if job_type == 'prepwizard':
    print('as a Preparation Wizard calculation.')
elif job_type == 'grid':
    print('as a Glide Grid calculation.')
elif job_type == 'docking':
    print('as a Glide Docking calculation.')

# Create function to parse log file
def checkLogFile(log_file, calculation_type):
    """
    Check wether calculations are succesful or failures according to the calculation
    type.

    Parameters
    ==========
    log_file : str
        Path to the calculation log file
    calculation_type : str
        Calculation type identifier
    """

    finished = False
    if calculation_type == 'prepwizard':
        finish_line = 'DONE. Output file:'
        error_line = 'ERROR'
    elif calculation_type == 'grid':
        finish_line = 'Exiting Glide'
        error_line = 'GLIDE FATAL ERROR:'
    elif calculation_type == 'docking':
        finish_line = 'Finished at:'
        error_line = 'GLIDE FATAL ERROR:'

    with open(log_file) as lf:
        for l in lf:
            if finish_line in l:
                finished = True
            elif error_line in l:
                if calculation_type == 'prepwizard':
                    raise ValueError('Prepartion wizard failed for model '+log_file.replace('.log',''))
                elif calculation_type == 'grid':
                    raise ValueError('Grid Glide calculation failed for model '+log_file.replace('.log',''))
                elif calculation_type == 'grid':
                    raise ValueError('Docking Glide calculation failed for model '+log_file.replace('.log',''))
    return finished

while True:
    finished = checkLogFile(log_file, job_type)
    if finished:
        if job_type == 'prepwizard':
            print('Preparation Wizard succeeded for model '+log_file.replace('.log','')+'\n')
        elif job_type == 'grid':
            print('Grid Glide calculation succeeded for model '+log_file.replace('.log','')+'\n')
        elif job_type == 'docking':
            print('Glide Docking succeeded for model '+log_file.replace('.log','')+'\n')
        break
    time.sleep(1)
