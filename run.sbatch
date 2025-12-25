#!/bin/bash
#SBATCH --job-name=AMS-ks21
#SBATCH --ntasks=64
#SBATCH --account=gts-rvuduc3
#SBATCH -qembers
#SBATCH --mem=0
#SBATCH --time=6:00:00

# Activate your virtual environment
module load mvapich2
source venv/bin/activate
./make.sh

# Run your distributed solver
srun dist-solve ~/scratch/instances/21/ks_21.cnf 21
