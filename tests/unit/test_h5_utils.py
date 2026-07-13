import numpy as np
from cap.utils.h5_utils import H5Store


def test_statistics_round_trip(tmp_path):
    stats = {
        "layer.0": {
            "t_stats": np.random.randn(768),
            "p_values": np.random.rand(768),
            "p_corrected": np.random.rand(768),
            "cohens_d": np.random.randn(768),
            "mean_diff": np.random.randn(768),
            "pooled_std": np.abs(np.random.randn(768)),
            "significant": np.zeros(768, dtype=bool),
        }
    }
    H5Store.save_statistics(statistics=stats, output_path=tmp_path)
    loaded = H5Store.load_statistics(tmp_path / "statistics.h5")
    assert np.allclose(stats["layer.0"]["t_stats"], loaded["layer.0"]["t_stats"])


def test_check_existing_data_missing(tmp_path):
    exists, meta = H5Store.check_existing_data(tmp_path)
    assert not exists
    assert meta is None
