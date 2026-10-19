#!/bin/bash -l

#SBATCH --job-name=ABC_purity_histogram
#SBATCH --output=logs/histogram_%j.out
#SBATCH --error=logs/histogram_%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# From the project directory:
#   mkdir -p logs
#   sbatch run_histogram_analysis.sh

set -eo pipefail

cd /home/wm-sharath/fib/superpoint_analysis

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpnet_l40s

INPUT_DIR="/data/users/wm-sharath/fib/ABC_final_geon_superpoints_reg_0.05/"

srun python -u histogram_analysis.py "$INPUT_DIR" "$@"
