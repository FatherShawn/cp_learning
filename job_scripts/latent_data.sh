#!/bin/bash
#SBATCH --job-name="latentData"
#SBATCH --partition production
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin latentData"

python $(pwd)/cp_learning/latent_reprocessor.py  --source_path $(pwd)/encoded --storage_path $(pwd)/labeled_encoded --filtered --reduction_factor 0.329032
