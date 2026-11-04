#!/bin/bash
#
# Starts a vLLM server on this GPU, waits until it is up, then runs the dev self-check
# (twice, for the reproducibility check) and the held-out evals. Results land in results/.
#
# If the TAs posted a shared endpoint instead, skip this script and run eval_dev.py with
# --endpoint <their URL> directly.
#
# Usage (from the repo root):
#   sbatch slurm/run_eval.sbatch
#   A3_MODEL=meta-llama/Llama-3.2-3B-Instruct sbatch slurm/run_eval.sbatch   # other model

#SBATCH --job-name=a3-eval
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -uo pipefail
module load anaconda3
conda activate "${A3_ENV:-CAI_A3}"

export HF_HOME=/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="${A3_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT=$((8000 + SLURM_JOB_ID % 1000))
EP="http://localhost:${PORT}/v1"
mkdir -p results slurm/logs

vllm serve "$MODEL" --port "$PORT" --max-model-len 8192 --seed 42 \
    > "slurm/logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "Waiting for vLLM ($MODEL) on port $PORT ..."
for _ in $(seq 1 120); do
    curl -sf "http://localhost:${PORT}/v1/models" > /dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died; see slurm/logs/vllm-${SLURM_JOB_ID}.log"; exit 1; }
    sleep 5
done
curl -sf "http://localhost:${PORT}/v1/models" > /dev/null || { echo "vLLM never came up"; exit 1; }
echo "vLLM is up."

python eval_dev.py --endpoint "$EP" --model "$MODEL" | tee results/dev_run1.txt
python eval_dev.py --endpoint "$EP" --model "$MODEL" | tee results/dev_run2.txt
python eval_extra/run_heldout.py --endpoint "$EP" --model "$MODEL" \
    --answers results/answers_heldout.jsonl | tee results/heldout.txt
python eval_extra/run_heldout.py --endpoint "$EP" --model "$MODEL" --config '{"rewrite":"off"}' \
    --answers results/answers_norewrite.jsonl | tee results/heldout_norewrite.txt

if diff -q <(grep -E "hit-rate|probes" results/dev_run1.txt) <(grep -E "hit-rate|probes" results/dev_run2.txt) > /dev/null; then
    echo "REPRODUCIBILITY: dev runs match"
else
    echo "REPRODUCIBILITY: dev runs DIFFER, check results/dev_run*.txt"
fi
