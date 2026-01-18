#!/bin/bash
#SBATCH -J training-pipeline
#SBATCH -A plgar2025-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu
#SBATCH -t 00:10:00
#SBATCH -o training-pipeline_%j.out
#SBATCH -e training-pipeline_%j.err


module purge
module load Python/3.10.4
module load CUDA/12.1.1

# activate environment
source /net/tscratch/people/plgjuliaryb/venvs/comp-lingu/bin/activate

cd ..
# python train.py
python run_baseline.py
