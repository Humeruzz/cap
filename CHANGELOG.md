# Changelog

All notable changes to this project are documented here. Versions from `0.1.0`
onward track the open-source restructure of the codebase.

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
