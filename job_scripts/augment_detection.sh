#!/bin/bash
#SBATCH --job-name="detectionAugment"
#SBATCH --partition production
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin detectionAugment"

python $(pwd)/cp_learning/detected.py  --source_path $(pwd)/CP_Quack-echo-2021-10-14-01-01-01.tar --storage_path $(pwd)/latent_prediction
