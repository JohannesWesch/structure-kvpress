#!/bin/bash
#SBATCH --job-name=kvpress-infinitebench
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
DATASET="infinitebench"
FRACTION=0.1

PRESS_NAMES=("kvsquared" "kvzip" "keydiff")
COMPRESSION_RATIOS=(0.9 0.95 0.98)
# longbook_sum_eng is excluded: unsupported by the scorer (rouge library conflict)
DATA_DIRS=(
  "passkey"
  "kv_retrieval"
  "number_string"
  "code_run"
  "code_debug"
  "math_find"
  "math_calc"
  "longbook_qa_eng"
  "longbook_qa_chn"
  "longbook_choice_eng"
  "longdialogue_qa_eng"
)

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
job_idx=0

for data_dir in "${DATA_DIRS[@]}"; do
  for ratio in "${COMPRESSION_RATIOS[@]}"; do
    for press in "${PRESS_NAMES[@]}"; do
      gpu_id=$((job_idx % NUM_GPUS))
      echo "Running press_name: $press, compression_ratio: $ratio, data_dir: $data_dir on GPU cuda:$gpu_id"
      (
        cd evaluation && python evaluate.py \
          --press_name "$press" \
          --compression_ratio "$ratio" \
          --model "$MODEL" \
          --dataset "$DATASET" \
          --data_dir "$data_dir" \
          --fraction "$FRACTION" \
          --device "cuda:$gpu_id"
      ) > "$LOG_DIR/${press}_${ratio}_${data_dir}.out" 2> "$LOG_DIR/${press}_${ratio}_${data_dir}.err" &

      job_idx=$((job_idx + 1))
      if (( job_idx % NUM_GPUS == 0 )); then
        wait
      fi
    done
  done
done

wait
echo "All evaluations completed."
