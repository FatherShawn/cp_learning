#!/bin/bash
#SBATCH --job-name="n-t-32-256"
#SBATCH --partition production
#SBATCH --nodes=24
#SBATCH --ntasks=192
#SBATCH --mem=32GBman

# pytorch lighting tune did not correctly report a working batch size for this model, so try several.

# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin batch-18-24-28"

echo "############# Batch size 18"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --accelerator cpu --num_nodes 24 --batch_size 18 --num_workers 8 --embed_size 96 --hidden_size 256 --fast_dev_run 10

echo "############# Batch size 24"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --accelerator cpu --num_nodes 24 --batch_size 24 --num_workers 8 --embed_size 96 --hidden_size 256 --fast_dev_run 10

echo "############# Batch size 28"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --accelerator cpu --num_nodes 24 --batch_size 28 --num_workers 8 --embed_size 96 --hidden_size 256 --fast_dev_run 10
