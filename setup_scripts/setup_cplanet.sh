#!/bin/bash
# Setup the environment in the HPC cluster.

# Load modules
echo "Restore modules"
module purge
module restore
echo "Activate cplanet conda environment"
# Running a subshell, we need to run the init hook code directly.
eval "$(conda shell.bash hook)"
if { conda env list | grep 'cplanet'; } >/dev/null 2>&1;
then
    echo "Updating cplanet environment"
    conda activate cplanet
    conda env update --file /global/u/shawn_bc_10/cplanet.yml
    echo "Environment cplanet updated and activated"
else
    conda env create --file /global/u/shawn_bc_10/cplanet.yml
    conda activate cplanet
    echo "Environment cplanet created and activated"
fi
# Python now at ~/.conda/envs/cplanet/bin/python
# These packages are not in conda and would not build with conda-build skeleton:
echo "Add non-conda packages"
pip install 'fairseq>=0.10.0'
pip install 'urlextract>=1.3.0'
pip install 'webdataset==0.1.62'
# Pull down project code
echo "Clone or update project code"
if [[ -d /scratch/shawn_bc_10/cp_learning ]]
then
    cd /scratch/shawn_bc_10/cp_learning
    git fetch && git pull
else
  cd /scratch/shawn_bc_10
  git clone https://github.com/FatherShawn/cp_learning.git
fi
