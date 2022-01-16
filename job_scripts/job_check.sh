#!/bin/bash
#SBATCH --job-name="demoJob"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin demoJob"

# actual binary (with IO redirections) and required input
# parameters is called in the next line
ls -lah

