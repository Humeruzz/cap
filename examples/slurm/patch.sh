#!/bin/bash
# Patch significant neurons and benchmark the patched model on a GPU node.
# Run this after capture.sh has produced runs/english/.
# Submit with:  sbatch examples/slurm/patch.sh
# Or chain it after capture:
#   jid=$(sbatch --parsable examples/slurm/capture.sh)
#   sbatch --dependency=afterok:$jid examples/slurm/patch.sh
#
# Copy this file and edit the two placeholders below before submitting.

#SBATCH --job-name=cap-patch
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION   # e.g. gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cap-patch-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REPLACE_WITH_YOUR_EMAIL       # e.g. you@example.org

set -euo pipefail

module load cuda                       # adjust to your cluster's module names
source /path/to/cap/.venv/bin/activate # needs the [lm-eval] extra for built-in benchmarks

OUT=runs/english

# Ablate (scale 0.0) the significant neurons and measure the effect on HellaSwag.
cap patch \
  --experiment "$OUT" \
  --benchmark hellaswag --n-eval-samples 200 \
  --scale 0.0

echo "Patch + benchmark complete. Results under $OUT/patching_results/"
