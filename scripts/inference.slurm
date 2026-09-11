#!/bin/bash
#SBATCH --job-name=tdbm-infer
#SBATCH --account=<your_allocation>
#SBATCH --partition=<your_gpu_partition>
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tdbm

mkdir -p logs
CHECKPOINT=${1:?usage: sbatch scripts/inference.slurm <checkpoint> <input_dir> <output_dir>}
INPUT=${2:?}
OUTPUT=${3:?}

python inference.py --config config/tdbm.json \
    --checkpoint "$CHECKPOINT" --input "$INPUT" --output "$OUTPUT" --mode multicrop
