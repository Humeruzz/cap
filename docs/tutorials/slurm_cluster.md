# Running on a SLURM Cluster

On a cluster you typically split the pipeline into stages: **capture** (needs the GPU),
**stats** (CPU-only, cheap), and **patch** (GPU again). Each is a separate `cap`
invocation, so each maps cleanly to a SLURM job.

!!! note
    `cap capture` already writes `statistics.h5`. Run `cap stats` as its own stage only to
    **recompute** statistics on an existing capture — for example with a different test.

## A single-job template

For a modest run, one GPU job that captures then patches is enough:

```bash
#!/bin/bash
#SBATCH --job-name=cap-en
#SBATCH --partition=gpu            # set to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cap-%j.log
# #SBATCH --mail-user=you@example.org   # uncomment + set to get email

set -euo pipefail
module load cuda                    # adjust to your cluster's module names
source /path/to/cap/.venv/bin/activate

OUT=runs/english
DATA="$(python -c "from importlib.resources import files; print(files('cap.data')/'files'/'english_smol.csv')")"

# Stage 1 — capture (GPU). Device auto-selects CUDA on a GPU node.
cap capture --model Qwen/Qwen3-8B --type faithfulness \
  --data "$DATA" --text-column srcs --label-column label \
  --output "$OUT"

# Stage 2 — patch + benchmark (GPU). Needs the lm-eval extra for built-ins.
cap patch --experiment "$OUT" --benchmark hellaswag --n-eval-samples 200 --scale 0.0
```

Submit it:

```bash
sbatch cap_english.sh
```

## Chaining stages as dependent jobs

The repo ships two ready templates under `examples/slurm/`: `capture.sh` (GPU) and
`patch.sh` (GPU). Chain them so patch starts only if capture succeeds:

```bash
jid=$(sbatch --parsable examples/slurm/capture.sh)
sbatch --dependency=afterok:$jid examples/slurm/patch.sh
```

To keep the cheap CPU stage off a GPU node, add a `cap stats` job on a CPU partition
(a copy of `capture.sh` that runs `cap stats` and drops `--gres`) and chain all three:

```bash
jid_cap=$(sbatch --parsable examples/slurm/capture.sh)                     # GPU
jid_sts=$(sbatch --parsable --dependency=afterok:$jid_cap stats.sh)        # CPU partition
sbatch --dependency=afterok:$jid_sts examples/slurm/patch.sh               # GPU
```

`stats.sh` needs no GPU (`cap stats` loads no model), so target a CPU partition and drop
`--gres`. Each stage reads the previous stage's output directory, so pass the same
`--experiment` / `--output` path throughout.

!!! tip
    The repo's `examples/slurm/` directory holds these templates (`capture.sh`, `patch.sh`).
    Treat them as examples: set `--partition`, `--gres`, and the `--mail-user` line for your
    environment before submitting.
