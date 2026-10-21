#!/bin/bash -l

#SBATCH --job-name=ABC_purity_histogram
#SBATCH --output=logs/histogram_%j.out
#SBATCH --error=logs/histogram_%j.err
#SBATCH --time=06:00:00
#SBATCH --qos=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# From the project directory:
#   mkdir -p logs
#   sbatch run_histogram_analysis.sh
#   sbatch run_histogram_analysis.sh --limit 100 --progress-every 20
#   sbatch --cpus-per-task=2 run_histogram_analysis.sh
#
# Four independent file workers, one native thread each; no GPU request.
# Six hours leaves headroom beyond the old one-hour timeout. Keep the previous
# 64 GB allocation: Slurm reported over 20 GB in the interrupted runs.
# PROJECT_DIR, CONDA_ENV, and INPUT_DIR can override the defaults below.
# Unreadable files are reported and excluded; analysis exits nonzero if any fail.
# Extra arguments are forwarded to histogram_analysis.py.

set -eo pipefail

cd "${PROJECT_DIR:-/home/wm-sharath/fib/superpoint_analysis}"
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_CPUS_PER_TASK:?Request CPUs with --cpus-per-task}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-hpnet_l40s}"
set -u

INPUT_DIR="${INPUT_DIR:-/mnt/gpu_data_storage/sharath/ABC/ABC_final_geon_superpoints_reg_0.025/}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_THREAD_LIMIT=1

echo "Job: ${SLURM_JOB_ID}; CPUs/workers: ${SLURM_CPUS_PER_TASK}; native threads per worker: 1"
echo "Conda environment: ${CONDA_DEFAULT_ENV}; input: ${INPUT_DIR}"

exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u histogram_analysis.py \
    "$INPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --skip-errors \
    "$@"
