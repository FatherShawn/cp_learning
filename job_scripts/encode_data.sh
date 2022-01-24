#!/bin/bash
#SBATCH --job-name="dataEncoder"
#SBATCH --partition production
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB


# change to the working directory
cd $SLURM_WORKDIR

echo ">>>> Begin dataEncoder"

# Copy and expand data
tar -xf ./encoder_data.tar --checkpoint=.1000
echo ">> Data ready for the run"
python ae_processor.py  --encode --filtered --data_dir $(pwd)/pickled --storage_path $(pwd)/encoded --batch_size 6 --num_workers 4 --embed_size 96 --hidden_size 256 --gpus 1 --checkpoint_path /global/u/shawn_bc_10/checkpoints/epoch-40-step-178853.ckpt
echo ">> Data ready for the run"
tar -cf $(pwd)/encoder_processed.tar $(pwd)//encoded
rm -r $(pwd)//encoded
echo ">> Encoded data ready"
