#!/bin/bash
# Capture activations (and the contrastive statistics) on a GPU node.
# Submit with:  sbatch examples/slurm/capture.sh
#
# Copy this file and edit the two placeholders below before submitting.

#SBATCH --job-name=cap-capture
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION   # e.g. gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cap-capture-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REPLACE_WITH_YOUR_EMAIL       # e.g. you@example.org

set -euo pipefail

module load cuda                       # adjust to your cluster's module names
source /path/to/cap/.venv/bin/activate # point at your CAP virtualenv

OUT=runs/english
DATA="$(python -c "from importlib.resources import files; print(files('cap.data')/'files'/'english_smol.csv')")"

# Device auto-selects CUDA on a GPU node; global flags go before the subcommand.
cap capture \
  --model Qwen/Qwen3-8B --type faithfulness \
  --data "$DATA" --text-column srcs --label-column label \
  --output "$OUT"

echo "Capture complete. Statistics + activations written to $OUT"
