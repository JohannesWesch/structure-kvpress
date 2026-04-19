#!/bin/bash
#SBATCH --job-name=kvpress-ruler-4k
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

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET="ruler"
DATA_DIR="4096"
FRACTION=1.0

# KVSquaredPress with different inner_press scorers (see evaluate_registry.py).
PRESS_NAMES=("kvsquared_knorm" "kvsquared_cur" "kvsquared_random" "kvsquared_keydiff" "kvsquared_streaming_llm")
COMPRESSION_RATIOS=(0.9 0.95 0.98)

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

declare -a GPU_PIDS
for ((i=0; i<NUM_GPUS; i++)); do
  GPU_PIDS[$i]=0
done

wait_for_gpu() {
  local gpu_id=$1
  local pid=${GPU_PIDS[$gpu_id]}
  if [[ $pid -ne 0 ]]; then
    wait "$pid"
  fi
}

assign_gpu() {
  while true; do
    for ((i=0; i<NUM_GPUS; i++)); do
      local pid=${GPU_PIDS[$i]}
      if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
        if [[ $pid -ne 0 ]]; then
          wait "$pid"
        fi
        echo "$i"
        return
      fi
    done
    sleep 2
  done
}

# Run no_press baseline (compression_ratio is overridden to 0.0 internally)
gpu_id=$(assign_gpu)
echo "Running no_press baseline on GPU cuda:$gpu_id"
(
  cd evaluation && python evaluate.py \
    --press_name "no_press" \
    --compression_ratio 0.0 \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --fraction "$FRACTION" \
    --device "cuda:$gpu_id"
) > "$LOG_DIR/no_press_0.0.out" 2> "$LOG_DIR/no_press_0.0.err" &
GPU_PIDS[$gpu_id]=$!

for ratio in "${COMPRESSION_RATIOS[@]}"; do
  for press in "${PRESS_NAMES[@]}"; do
    gpu_id=$(assign_gpu)
    echo "Running press_name: $press, compression_ratio: $ratio on GPU cuda:$gpu_id"
    (
      cd evaluation && python evaluate.py \
        --press_name "$press" \
        --compression_ratio "$ratio" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --data_dir "$DATA_DIR" \
        --fraction "$FRACTION" \
        --device "cuda:$gpu_id"
    ) > "$LOG_DIR/${press}_${ratio}.out" 2> "$LOG_DIR/${press}_${ratio}.err" &
    GPU_PIDS[$gpu_id]=$!
  done
done

wait
echo "All evaluations completed."
