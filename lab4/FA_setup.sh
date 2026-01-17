#!/bin/bash
#SBATCH -J flash_attn_check
#SBATCH -A plgar2025-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --tasks-per-node=4      # max P we'll use below
#SBATCH --gres=gpu
#SBATCH -t 00:10:00
#SBATCH --output=flash_attn_%j.log
#SBATCH --error=flash_attn_%j.err

set -e # exit the script if anything fails

module purge
module load Python/3.10.4
module load CUDA/12.1.1

# activate environment
source /net/tscratch/people/plgjuliaryb/venvs/comp-lingu/bin/activate

# # diagnostics
# python - <<'EOF'
# import torch
# print("torch:", torch.__version__)
# print("torch cuda build:", torch.version.cuda)
# print("cuda available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("gpu:", torch.cuda.get_device_name(0))
# EOF

# nvcc -V
# nvidia-smi

# # from FlashAttention docs: https://github.com/Dao-AILab/flash-attention
# pip install packaging
# pip install ninja
# ninja --version
# echo $? # exit code of the last command executed (should be 0 for correct ninja installation)


# MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir

python - <<'EOF'
import flash_attn
print("flash-attn version:", flash_attn.__version__)
EOF
