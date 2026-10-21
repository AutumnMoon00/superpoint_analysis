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
#   sbatch run_histogram_analysis_2.sh
#   sbatch run_histogram_analysis_2.sh --limit 100 --progress-every 20
# Uses the shared launcher with the 0.05 input directory.
# Worker count follows --cpus-per-task; each worker uses one native thread.

set -eo pipefail

export PROJECT_DIR="${PROJECT_DIR:-/home/wm-sharath/fib/superpoint_analysis}"
export INPUT_DIR="${INPUT_DIR:-/mnt/gpu_data_storage/sharath/ABC/ABC_final_geon_superpoints_reg_0.05/}"
exec bash "$PROJECT_DIR/run_histogram_analysis.sh" "$@"
