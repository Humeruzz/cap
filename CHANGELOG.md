# Changelog

All notable changes to this project are documented here. Versions from `0.1.0`
onward track the open-source restructure of the codebase.

## 1.0.1 — 2026-08-20

Maintenance release: bug fixes, CLI ergonomics, and a reproducible lint toolchain.

- Fixed HellaSwag evaluation: the benchmark now loads from `Rowan/hellaswag`, as
  the bare `hellaswag` dataset id no longer resolves on the Hugging Face Hub.
- Fixed non-ASCII rendering in the stats viewer, which was missing a
  `<meta charset="UTF-8">` declaration.
- `cap visualize --experiments` now accepts comma-separated paths
  (`--experiments A,B`) alongside the repeated flag, matching `--downsample`.
  Previously `A,B` was read as a single path and reported as "requires at least
  two" even though two were given.
- `cap visualize --mode similarity` now reports a clear error when an experiment
  directory has no `statistics.h5`, instead of raising `AttributeError`.
- Fixed mutable list defaults on `PatchExperiment.run()`, which were shared
  across calls.
- Documented steering: a new Concepts → Steering section, and `cap patch --scale`
  now explains ablation (`0.0`) versus steering (a factor near `1`).
- Reframed the README and docs around contrastive activation patching for any
  target behaviour; faithfulness and factuality are the running example rather
  than the tool's purpose.
- Pinned ruff and declared the lint rule set explicitly, so a ruff release can no
  longer change what CI enforces.

## 1.0.0 — 2026-07-03

First stable release.

- Moved `paper/` research scripts under `examples/paper/` and removed the legacy
  cluster-specific `scripts/` directory (superseded by `examples/slurm/`).
- Renamed `cap.plot.plot_stats` → `cap.plot.stats_html` and
  `cap.plot.plot_similarity` → `cap.plot.similarity_html`; the old module paths
  remain as deprecation shims and will be removed in a future release.
- Removed dead code: `cap.utils.qwen_inference` and the `cap.utils.data_utils`
  deprecation shim.
- Added `cap --version`.
- Added `CITATION.cff` and a GitHub Release workflow that builds and uploads
  distributions on `v*` tags.

## 0.4.0

- Added the MkDocs documentation site (`mkdocs build --strict`, 14 pages).
- Fixed the CLI to the single-command form (`cap run`, not `cap run run`) and
  corrected packaging so all subpackages ship in the wheel.
- Added five runnable example scripts, the `examples/slurm/` templates, and the
  bundled `quickstart_synthetic.csv` dataset.

## 0.3.0

- Added type annotations across `src/cap/` so `mypy src/cap/` passes.
- Rebuilt the test suite: deterministic unit tests plus a gpt2 CPU integration
  smoke test; raised core/data coverage to 96%.

## 0.2.0

- Added the labeled-data interface (`LabeledDataset`, `TwoGroupDataset`),
  `PromptTemplate` / `PROMPT_TYPES`, and the evaluator interface
  (`EvaluatorProtocol`, `CSVEvaluator`, `BUILTIN_EVALUATORS`).
- Wired the full CLI (`capture`, `stats`, `patch`, `visualize`, `run`).

## 0.1.0

- Foundation: `src/` package layout, `pyproject.toml`, and the CLI skeleton.
- Removed cluster-specific dead code and default paths.

## 0.1.0a1 — 2026-04-23 (pre-release)

First pre-release. Scoped for internal sharing with fellow students.

- Removed cluster-specific default paths; `--output_dir` (and `--model` / `--h5_path` where applicable) are now required CLI arguments.
- Templatized SLURM scripts under `scripts/` (user email commented out, set your own before use).
- Added `LICENSE` (MIT).
- Exposed `cap.__version__`.
