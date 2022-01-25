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

python $(pwd)/ae_processor.py  --data_dir /scratch/shawn_bc_10/pickled --batch_size 12 --num_workers 8 --embed_size 96 --hidden_size 256 --gpus 8 --checkpoint_path /global/u/shawn_bc_10/checkpoints/epoch-40-step-178853.ckpt
