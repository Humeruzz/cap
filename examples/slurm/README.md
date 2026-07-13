# SLURM example scripts

Templates for running CAP on a SLURM cluster. They split the pipeline into the stages
that actually need different resources:

- **`capture.sh`** — capture activations (needs the GPU). Writes `activations.h5` +
  `statistics.h5` into an output directory.
- **`patch.sh`** — patch the significant neurons and benchmark the patched model
  (needs the GPU and the `[lm-eval]` extra).

> `cap capture` already writes `statistics.h5`, so there is no separate stats job here.
> Run `cap stats` as its own (CPU-only) stage only to *recompute* statistics on an
> existing capture — for example with a different test.

## Before you submit

Copy a script and replace **both** placeholders in each file:

- `# REPLACE_WITH_YOUR_PARTITION` — your cluster's GPU partition name (the `--partition` line)
- `# REPLACE_WITH_YOUR_EMAIL` — your address for job notifications (the `--mail-user` line)

Also edit the two environment lines to match your cluster:

```bash
module load cuda                        # your CUDA / toolchain module name
source /path/to/cap/.venv/bin/activate  # path to your CAP virtualenv
```

## Submit

Run capture, then patch:

```bash
sbatch examples/slurm/capture.sh
sbatch examples/slurm/patch.sh          # after capture.sh finishes
```

Or chain them so patch starts automatically only if capture succeeds:

```bash
jid=$(sbatch --parsable examples/slurm/capture.sh)
sbatch --dependency=afterok:$jid examples/slurm/patch.sh
```

To keep the cheap CPU work off a GPU node, you can additionally split out a `cap stats`
job onto a CPU partition (drop `--gres`) and chain capture → stats → patch with
`--dependency=afterok:` between each. See the *Running on a SLURM Cluster* tutorial in the
docs for the three-job pattern.
