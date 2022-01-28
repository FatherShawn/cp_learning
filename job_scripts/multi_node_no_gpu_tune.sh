#!/bin/bash
#SBATCH --job-name="batchCheck"
#SBATCH --partition development
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --mem=8GB

# pytorch lighting tune did not correctly report a working batch size for this model, so try several.

# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin batchCheck"

echo "############# Batch size 12"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --batch_size 12 --num_workers 8 --embed_size 96 --hidden_size 256 --checkpoint_path $(pwd)/archived-checkpoints/epoch-40-step-178853.ckpt --fast_dev_run 10

echo "############# Batch size 24"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --batch_size 24 --num_workers 8 --embed_size 96 --hidden_size 256 --checkpoint_path $(pwd)/archived-checkpoints/epoch-40-step-178853.ckpt --fast_dev_run 10

echo "############# Batch size 48"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --batch_size 48 --num_workers 8 --embed_size 96 --hidden_size 256 --checkpoint_path $(pwd)/archived-checkpoints/epoch-40-step-178853.ckpt --fast_dev_run 10
