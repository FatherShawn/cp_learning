#!/bin/bash
#SBATCH --job-name="18-batch"
#SBATCH --partition production
#SBATCH --nodes=16
#SBATCH --ntasks=128
#SBATCH --mem=32Gb

# pytorch lighting tune did not correctly report a working batch size for this model, so try several.

# change to the working directory
cd $SLURM_WORKDIR

#echo "NODELIST=${SLURM_NODELIST}"
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
#echo "MASTER_ADDR=${MASTER_ADDR}"
#export MASTER_PORT=95328
#echo "MASTER_PORT=${MASTER_PORT}"

echo ">>>> Begin batch 18"

srun python $(pwd)/cp_learning/ae_processor.py  --exp_label "autoencoder batch tune cpu [18]" --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --accelerator cpu --num_nodes 16 --batch_size 18 --num_workers 7 --embed_size 96 --hidden_size 256 --max_epochs 1 --limit_train_batches 56 --limit_val_batches 56
