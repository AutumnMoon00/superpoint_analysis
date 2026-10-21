#!/bin/bash -l

#SBATCH --job-name=ABC_combine_superpoints
#SBATCH --output=logs/combine_superpoints_%j.out
#SBATCH --error=logs/combine_superpoints_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

# Submit from /home/wm-sharath/fib/superpoint_analysis:
#   mkdir -p logs
#   sbatch run_combine_superpoints.sh --limit 20 --progress-every 5
#   sbatch run_combine_superpoints.sh
#   sbatch --cpus-per-task=2 run_combine_superpoints.sh
#   sbatch --cpus-per-task=1 run_combine_superpoints.sh
#
# Four independent file workers by default; one native thread per worker.
# Requests 4 GB RAM for the whole job and no GPU. Worker count follows CPUs.
# The sequential test used about 150 MB; 4 GB leaves headroom for four workers.
# Start with four workers: network storage may limit scaling at higher counts.
# Allow 48 hours for both formats (about 64,000 files) over network storage.
# research QoS permits up to 72 hours; adjust --time at submission if needed.
# Existing completed outputs are skipped when the job is run again.
# PROJECT_DIR, CONDA_ENV, and DATA_DIR can override the defaults below.
# All command-line arguments are forwarded to combine_superpoints.py.
# New output directories and timestamped CSV reports use Python's defaults.

set -eo pipefail

# Slurm runs a spooled copy; do not derive the project path from this script.
cd "${PROJECT_DIR:-/home/wm-sharath/fib/superpoint_analysis}"
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_CPUS_PER_TASK:?Request CPUs with --cpus-per-task}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-hpnet_l40s}"
set -u

DATA_DIR="${DATA_DIR:-/mnt/gpu_data_storage/sharath/ABC}"

# Each file worker uses only one native numerical thread.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_THREAD_LIMIT=1

echo "Job: ${SLURM_JOB_ID}; CPUs/workers: ${SLURM_CPUS_PER_TASK}; native threads per worker: 1"
echo "Conda environment: ${CONDA_DEFAULT_ENV}; data directory: ${DATA_DIR}"

exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u combine_superpoints.py \
    --data-dir "$DATA_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    "$@"
