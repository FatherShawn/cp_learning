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
echo "SLURM Node List: ${SLURM_NODELIST}"

# Process the node list
NODE_LIST=$( scontrol show hostname $SLURM_JOB_NODELIST | sed -z 's/\n/\:4,/g' )
NODE_LIST=${NODE_LIST%?}
echo "Processed Node list: ${NODE_LIST}"

horovodrun -np 4 -H $NODE_LIST python $(pwd)/cp_learning/ae_processor_horovod.py  --exp_label "autoencoder tune - lr: 0.01->0.0001 max 100" \
--data_dir $(pwd)/pickled \
--comet_storage $(pwd)/comet_storage_tune \
--accelerator cpu \
--batch_size 4 \
--ray_nodes 4 \
--num_workers 4 \
--embed_size 96 \
--hidden_size 256 \
--l_rate 0.01 \
--l_rate_min 0.0001 \
--l_rate_max_epoch 100 \
--limit_train_batches 250 --limit_val_batches 250