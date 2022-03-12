#!/bin/bash
#SBATCH --job-name="latentData"
#SBATCH --partition production
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=44GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin latentData"

python $(pwd)/cp_learning/candidate_analyzer.py  --storage_path /scratch/shawn_bc_10/latent_prediction
