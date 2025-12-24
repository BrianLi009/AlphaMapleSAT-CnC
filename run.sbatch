#!/bin/bash
#SBATCH --job-name=AMS-ks19
#SBATCH --ntasks=384
#SBATCH --account=gts-rvuduc3
#SBATCH -qembers
#SBATCH --mem=0
#SBATCH --time=6:00:00

# Activate your virtual environment
source bin/activate

# Run your distributed solver
srun dist-solve ~/scratch/instances/ks_19.cnf 19 51
