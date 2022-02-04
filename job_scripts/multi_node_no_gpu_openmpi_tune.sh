#!/bin/bash
#SBATCH --job-name="18-batch"
#SBATCH --partition production
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --tasks-per-node=8
#SBATCH --mem=40Gb

# Auto resubmit.
#SBATCH --signal=SIGUSR1@90

# change to the working directory
cd $SLURM_WORKDIR
echo "nodes: ${SLURM_NODELIST}"

python $(pwd)/cp_learning/ae_processor_mpi.py  --exp_label "autoencoder lr: 0.005->0.0001 max 100" \
--run_id "mpi tune 4-32-40" \
--data_dir $(pwd)/pickled \
--comet_storage $(pwd)/comet_storage_tune \
--accelerator cpu \
--batch_size 4 \
--num_workers 7 \
--embed_size 96 \
--hidden_size 256 \
--l_rate 0.01 \
--l_rate_min 0.0001 \
--l_rate_max_epoch 100 \
--limit_train_batches 250 --limit_val_batches 250