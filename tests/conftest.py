import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_activations():
    """3 layers × 5 samples × 768 dims — no model needed."""
    return {f"layer.{i}.mlp": np.random.randn(5, 768) for i in range(3)}


@pytest.fixture
def tmp_csv(tmp_path):
    """10-row CSV with text + label columns (standardised schema: 'srcs', 'label')."""
    df = pd.DataFrame(
        {
            "srcs": [f"sentence {i}" for i in range(10)],
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tmp_experiment_dir(tmp_path):
    return tmp_path / "experiment"
