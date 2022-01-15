#!/bin/bash
#SBATCH --partition interactive
#SBATCH -J spdDeviceQuery
#SBATCH --ntasks l
#SBATCH --gres gpu:1
#SBATCH --mem <fond color="red">4</fond color>


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin spdDeviceQuery"

# actual binary (with IO redirections) and required input
# parameters is called in the next line
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery/
make clean
make

