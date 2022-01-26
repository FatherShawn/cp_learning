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
# Check data
python $(pwd)/cp_learning//dataset_test.py  --data_dir $(pwd)/pickled
