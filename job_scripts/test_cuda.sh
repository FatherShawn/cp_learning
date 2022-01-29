#!/bin/bash
#SBATCH --job-name="cudaVerifcation"
#SBATCH --partition development
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin cudaVerifcation"


# Check cuda
python $(pwd)/cp_learning/cuda_check.py
