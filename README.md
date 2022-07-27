# AQmortality
PhD project working repo 

## Environment setup
1. Run the setup.sh bash script
2. Activate the conda environment: `conda activate AQmort`
3. Login to weights and biases and follow the subsequent instructions: `wandb login`
4. Install the PyTorch libraries on JASMIN:
  - `conda install pytorch==1.10.0 cpuonly -c pytorch`
  - `module load gcc/8.2.0`
  - `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html`

Note: If not using JASMIN, different versions of pytorch and the torch libraries will need to be installed depending on the hardware being used. Check the PyTorch and PyTorch Geometric offical websites to determine the correct installation commands.
