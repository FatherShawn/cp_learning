#!/bin/bash
#SBATCH --job-name="demoJob"
#SBATCH --partition production
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin demoJob"

# actual binary (with IO redirections) and required input
# parameters is called in the next line
module list
conda env list
