#!/bin/bash
#
#SBATCH --job-name=voxel_models # Job name for tracking
#SBATCH --partition=falcon    # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4    # Number of CPU threads used by your job
#SBATCH --gres=gpu:1 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_50 # Events to send email on, remove if you don't want this
#SBATCH --output=job_output_%j.out # Standard out from your job
#SBATCH --error=job-output_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

# object is set to None, modify it you'd like to generate specific objects

## Execute your program(s) ##
module load CUDA
srun -u python3.9 sampling.py --epochs 502 --device 'cuda' --sched 'cosine' --timesteps 500 --s_type 'model' --loss 'hybrid' --object 'bathtub'

