#!/bin/bash
#SBATCH --job-name=tdbm-train
#SBATCH --account=<your_allocation>
#SBATCH --partition=<your_gpu_partition>
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x-%j.out

# Adjust to your cluster's module/conda setup.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tdbm

mkdir -p logs
python train.py --config config/tdbm.json
