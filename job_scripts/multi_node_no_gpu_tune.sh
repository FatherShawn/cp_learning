#!/bin/bash
#SBATCH --job-name="18-batch"
#SBATCH --partition production
#SBATCH --nodes=16
#SBATCH --ntasks=128
#SBATCH --mem=32Gb

# pytorch lighting tune did not correctly report a working batch size for this model, so try several.

# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin batch 18"

python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --accelerator cpu --num_nodes 16 --batch_size 18 --num_workers 7 --embed_size 96 --hidden_size 256 --fast_dev_run 10
