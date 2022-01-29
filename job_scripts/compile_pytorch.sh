#!/bin/bash
#SBATCH --job-name="cudaCompile"
#SBATCH --partition production
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin cudaCompile"

export CMAKE_PREFIX_PATH="/scratch/shawn_bc_10/.venv-1.8"
cd $(pwd)/pytorch
python setup.py installmu