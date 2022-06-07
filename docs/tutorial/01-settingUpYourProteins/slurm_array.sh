#!/bin/bash
#SBATCH --job-name=AF_sequences
#SBATCH --qos=bsc_ls
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --array=1-1
#SBATCH --output=AF_sequences_%a_%A.out
#SBATCH --error=AF_sequences_%a_%A.err

if [[ $SLURM_ARRAY_TASK_ID = 1 ]]; then
jobs
fi

