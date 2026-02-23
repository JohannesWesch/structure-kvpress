#!/bin/bash
#SBATCH --job-name=kvpress-eval
#SBATCH --partition=gpu_h100_il
#SBATCH --mem=510000mb
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/kvpress_eval_%j.out
#SBATCH --error=logs/kvpress_eval_%j.err

mkdir -p logs

module load devel/cuda/12.8

source .venv/bin/activate

PRESS_NAMES=("kvsquared" "kvsquared_2" "kvsquared_3" "kvsquared_4" "kvsquared_5" "kvsquared+" "kvsquared_2+" "kvsquared_3+" "kvsquared_4+" "kvsquared_5+")

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

for i in "${!PRESS_NAMES[@]}"; do
  press="${PRESS_NAMES[$i]}"
  gpu_id=$((i % NUM_GPUS))
  echo "Running press_name: $press on GPU cuda:$gpu_id"
  (
    cd evaluation && python evaluate.py --press_name "$press" --device "cuda:$gpu_id"
  ) > "logs/${press}_${SLURM_JOB_ID}.out" 2> "logs/${press}_${SLURM_JOB_ID}.err" &

  if (( (i + 1) % NUM_GPUS == 0 )); then
    wait
  fi
done

wait
echo "All evaluations completed."
