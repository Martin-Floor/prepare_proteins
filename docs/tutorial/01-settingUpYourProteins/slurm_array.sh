#!/bin/bash
#SBATCH --job-name=AF_sequences
#SBATCH --qos=bsc_ls
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres gpu:2
#SBATCH --ntasks=1
#SBATCH --array=1-1
#SBATCH --constraint=k80
#SBATCH --output=AF_sequences_%a_%A.out
#SBATCH --error=AF_sequences_%a_%A.err

module purge
module load singularity
module load alphafold

if [[ $SLURM_ARRAY_TASK_ID = 1 ]]; then
cd alphafold
Path=$(pwd)
bsc_alphafold --fasta_paths $Path/input_sequences/GPX-Bacillus-subtilis.fasta --output_dir=$Path/output_models --model_preset=monomer_ptm --max_template_date=2022-01-01 --random_seed 1
cd ..
fi

