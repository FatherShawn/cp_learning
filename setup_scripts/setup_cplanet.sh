#!/bin/bash
# Setup the environment in the HPC cluster.

# Load modules
module purge
module restore
conda env create --file /global/u/shawn_bc_10/cplanet.yml
conda activate cplanet
# Python now at ~/.conda/envs/cplanet/bin/python
# These packages are not in conda and would not build with conda-build skeleton:
pip install 'fairseq>=0.10.0'
pip install 'urlextract>=1.3.0'
pip install 'webdataset==0.1.62'
# Pull down project code
git clone https://github.com/FatherShawn/cp_learning.git