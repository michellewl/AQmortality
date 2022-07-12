#!/bin/bash

# Direct bash to where the conda command file is kept.
# source /opt/anaconda/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh

# Create the predefined conda environment with required libraries
conda env create -f AQmort_env.yml

# Set up the environment for use with the JASMIN Jupyter Notebook Service
conda install -y -n AQmort ipykernel
conda run -n AQmort python -m ipykernel install --user --name AQmort

