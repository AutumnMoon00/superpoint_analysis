#!/bin/bash -l

#SBATCH --job-name=ABC_superpoints_regs
#SBATCH --output=logs/abc_superpoints_regs_%j.out
#SBATCH --error=logs/abc_superpoints_regs_%j.err
#SBATCH --time=06:00:00
#SBATCH --qos=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Submit from /home/wm-sharath/fib/superpoint_analysis after creating logs:
#   mkdir -p logs
#   sbatch run_abc_superpoints.sh
# Both Python scripts live together in this project directory.
# Use PROJECT_DIR to override /home/wm-sharath/fib/superpoint_analysis.
# Runs 0.025, 0.05, and 0.1 sequentially by default, using all requested CPUs.
# Extra arguments are forwarded to diff_regularizations.py:
#   sbatch run_abc_superpoints.sh --limit 10 --progress-every 1
#   sbatch run_abc_superpoints.sh --regularizations 0.025 0.05 0.1
# Use CONDA_ENV to select another environment with geon, numpy and h5py.
# ABC_OUTPUT_DIR and GEON_OUTPUT_DIR are base paths; Python appends _reg_<value>.
# Output filenames stay unchanged; each regularization gets its own CSV report.

set -eo pipefail

# Slurm runs a spooled copy, so BASH_SOURCE does not identify the repo path.
cd "${PROJECT_DIR:-/home/wm-sharath/fib/superpoint_analysis}"
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_CPUS_PER_TASK:?Request CPUs with --cpus-per-task}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-hpnet_l40s}"
set -u

INPUT_DIR="${INPUT_DIR:-/data/users/wm-sharath/fib/ABC_final/}"
# ABC_OUTPUT_DIR="${ABC_OUTPUT_DIR:-/data/users/wm-sharath/fib/ABC_final_abc_superpoints/}"
# GEON_OUTPUT_DIR="${GEON_OUTPUT_DIR:-/data/users/wm-sharath/fib/ABC_final_geon_superpoints/}"

ABC_OUTPUT_DIR="${ABC_OUTPUT_DIR:-/mnt/gpu_data_storage/sharath/ABC/ABC_final_abc_superpoints/}"
GEON_OUTPUT_DIR="${GEON_OUTPUT_DIR:-/mnt/gpu_data_storage/sharath/ABC/ABC_final_geon_superpoints/}"


# Each Python worker uses one native thread; parallelism is across files.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

echo "Job: ${SLURM_JOB_ID}; CPUs/workers: ${SLURM_CPUS_PER_TASK}"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"

srun python -u diff_regularizations.py \
    --input-dir "$INPUT_DIR" \
    --abc-output-dir "$ABC_OUTPUT_DIR" \
    --geon-output-dir "$GEON_OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    "$@"
