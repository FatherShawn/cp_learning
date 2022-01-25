#!/bin/bash
#SBATCH --job-name="dataVerifcation"
#SBATCH --partition production
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin dataVerifcation"

# Copy and expand data
python $(pwd)/dataset_test.py  --encode --filtered --data_dir /scratch/shawn_bc_10/pickled
