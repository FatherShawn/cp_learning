#!/bin/bash
#SBATCH --job-name="imageConversion"
#SBATCH --partition production
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin imageConversion"

python $(pwd)/cp_learning/cp_image_processor.py  --source_path $(pwd)/pickled --storage_path $(pwd)/as_images --filtered
