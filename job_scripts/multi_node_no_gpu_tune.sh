#!/bin/bash
#SBATCH --job-name="18-batch"
#SBATCH --partition production
#SBATCH --nodes=5
#SBATCH --ntasks=40
#SBATCH --tasks-per-node=8
#SBATCH --mem=32Gb

# pytorch lighting tune did not correctly report a working batch size for this model, so try several.

# change to the working directory
cd $SLURM_WORKDIR
echo "nodes: ${SLURM_NODELIST}"

# __doc_head_address_start__
redis_password = $(uuidgen)
export redis_password

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --redis-password=${redis_password} \
    --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus 0 --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --redis-password=${redis_password} \
        --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus 0 --block &
    sleep 5
done
# __doc_worker_ray_end__

# __doc_script_start__
# ray/doc/source/cluster/examples/simple-trainer.py

echo ">>>> Begin batch 4 on 8/7"

python $(pwd)/cp_learning/ae_processor.py  --exp_label "autoencoder tune - 5 ray workers/7 io workers" --data_dir $(pwd)/pickled --comet_storage $(pwd)/comet_storage_tune --accelerator cpu --batch_size 4 --ray_nodes 5 --num_workers 7 --embed_size 96 --hidden_size 256 --max_epochs 1 --limit_train_batches 250 --limit_val_batches 250
