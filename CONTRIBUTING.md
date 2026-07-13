# Contributing

## Setup

```bash
git clone <repo-url>
cd cap
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest tests/unit/ -x -q
```

## Linting

```bash
ruff check src/cap/
ruff format src/cap/
```

## Integration tests (requires GPU or will be slow on CPU)

```bash
pytest tests/integration/ -m integration -v
```
