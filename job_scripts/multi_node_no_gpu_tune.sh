#!/bin/bash
#SBATCH --job-name="batch-0-16-128-16"
#SBATCH --partition development
#SBATCH --nodes=16
#SBATCH --ntasks=128
#SBATCH --mem=32GB

# pytorch lighting tune did not correctly report a working batch size for this model, so try several.

# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin batch-0-16-128-16"

echo "############# Batch size 6"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --accelerator cpu --num_nodes 16 --batch_size 6 --num_workers 8 --embed_size 96 --hidden_size 256 --fast_dev_run 10

echo "############# Batch size 8"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --accelerator cpu --num_nodes 16 --batch_size 8 --num_workers 8 --embed_size 96 --hidden_size 256 --gpus 1 --fast_dev_run 10

echo "############# Batch size 12"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --accelerator cpu --num_nodes 16 --batch_size 12 --num_workers 8 --embed_size 96 --hidden_size 256 --gpus 1 --fast_dev_run 10
