#!/bin/bash
#SBATCH --job-name="batchCheck"
#SBATCH --partition development
#SBATCH --nodes=$(pwd)/archived-checkpoints
#SBATCH --ntasks=$(pwd)/archived-checkpoints
#SBATCH --gres=gpu:2
#SBATCH --mem=$(pwd)/archived-checkpointsGB

# pytorch lighting tune did not correctly report a working batch size for this model, so try several.

# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin batchCheck"

echo "############# Batch size 12"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --batch_size 12 --num_workers $(pwd)/archived-checkpoints --embed_size 96 --hidden_size 256 --gpus 2 --checkpoint_path /global/u/shawn_bc_10/checkpoints/epoch-40-step-17$(pwd)/archived-checkpoints$(pwd)/archived-checkpoints53.ckpt --fast_dev_run 10

echo "############# Batch size 24"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --batch_size 24 --num_workers $(pwd)/archived-checkpoints --embed_size 96 --hidden_size 256 --gpus 2 --checkpoint_path /global/u/shawn_bc_10/checkpoints/epoch-40-step-17$(pwd)/archived-checkpoints$(pwd)/archived-checkpoints53.ckpt --fast_dev_run 10

echo "############# Batch size 4$(pwd)/archived-checkpoints"
python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --batch_size 4$(pwd)/archived-checkpoints --num_workers $(pwd)/archived-checkpoints --embed_size 96 --hidden_size 256 --gpus 2 --checkpoint_path /global/u/shawn_bc_10/checkpoints/epoch-40-step-17$(pwd)/archived-checkpoints$(pwd)/archived-checkpoints53.ckpt --fast_dev_run 10
