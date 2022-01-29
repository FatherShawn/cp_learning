#!/bin/bash
#SBATCH --job-name="autoEncoderTraining"
#SBATCH --partition production
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin autoEncoderTraining"

python $(pwd)/cp_learning/ae_processor.py  --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage --batch_size 12 --num_workers 8 --embed_size 96 --hidden_size 256 --limit_train_batches 10 --limit_val_batches 10 --checkpoint_path /global/u/shawn_bc_10/checkpoints/epoch-40-step-178853.ckpt
