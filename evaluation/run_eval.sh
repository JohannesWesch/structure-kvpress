#!/bin/bash
#SBATCH --job-name=kvpress-eval
#SBATCH --partition=gpu_h100_il
#SBATCH --mem=510000mb
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/%j/sbatch.out
#SBATCH --error=logs/%j/sbatch.err

LOG_DIR="logs/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

module load devel/cuda/12.8

source .venv/bin/activate

PRESS_NAMES=("kvzip" "kvsquared_2")
COMPRESSION_RATIOS=(0.25 0.5 0.75 0.9 0.95 0.98 0.99)

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
job_idx=0

for ratio in "${COMPRESSION_RATIOS[@]}"; do
  for press in "${PRESS_NAMES[@]}"; do
    gpu_id=$((job_idx % NUM_GPUS))
    echo "Running press_name: $press, compression_ratio: $ratio on GPU cuda:$gpu_id"
    (
      cd evaluation && python evaluate.py --press_name "$press" --compression_ratio "$ratio" --device "cuda:$gpu_id"
    ) > "$LOG_DIR/${press}_${ratio}.out" 2> "$LOG_DIR/${press}_${ratio}.err" &

    job_idx=$((job_idx + 1))
    if (( job_idx % NUM_GPUS == 0 )); then
      wait
    fi
  done
done

wait
echo "All evaluations completed."
