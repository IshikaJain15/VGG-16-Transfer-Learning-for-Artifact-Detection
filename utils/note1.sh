#!/bin/bash

#SBATCH --account=intro_vsc36405
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --clusters=genius
#SBATCH --ntasks=9
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=output1.log
#SBATCH --error=error1.log

# Load necessary modules (if required by your cluster)
module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

# Create a virtual environment and activate it
source ${VSC_DATA}/Automate-Image-Annotation-for-Decision-Support-Systems/Pytorch-UNet-master/myenv/bin/activate

# Install required packages
python -m pip install --upgrade pip
pip install -r requirements.txt


# Start Jupyter Notebook
python train.py
python evaluate.py
python predict.py


