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
echo "Activate cplanet conda environment"
# Running a subshell, we need to run the init hook code directly.
eval "$(conda shell.bash hook)"
if { conda env list | grep 'cplanet'; } >/dev/null 2>&1;
then
   conda activate cplanet
   echo "Environment cplanet activated"
else
  echo "Environment cplanet not found, exiting :("
  exit 1
fi
# Copy and expand data
python $(pwd)/cp_learning//dataset_test.py  --data_dir $(pwd)/pickled
