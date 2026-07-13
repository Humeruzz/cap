import pytest
from cap.data.loaders import LabeledDataset, TwoGroupDataset


def test_labeled_dataset_split(tmp_csv):
    ds = LabeledDataset.from_csv(tmp_csv, text_col="srcs", label_col="label")
    a, b = ds.as_groups()
    assert len(a) == 5  # label == 1
    assert len(b) == 5  # label == 0
    assert all(isinstance(t, str) for t in a)


def test_labeled_dataset_n_samples(tmp_csv):
    ds = LabeledDataset.from_csv(tmp_csv, text_col="srcs", label_col="label", n_samples=3)
    a, b = ds.as_groups()
    assert len(a) <= 3
    assert len(b) <= 3


def test_labeled_dataset_missing_column(tmp_csv):
    with pytest.raises(ValueError, match="text_col"):
        LabeledDataset.from_csv(tmp_csv, text_col="nonexistent", label_col="label")


def test_two_group_dataset(tmp_csv):
    ds = TwoGroupDataset.from_csv_pair(tmp_csv, tmp_csv, text_col="srcs")
    a, b = ds.as_groups()
    assert len(a) == 10
    assert len(b) == 10


def test_two_group_missing_column(tmp_csv):
    with pytest.raises(ValueError, match="text_col"):
        TwoGroupDataset.from_csv_pair(tmp_csv, tmp_csv, text_col="bad_col")


def test_labeled_dataset_missing_label_column(tmp_csv):
    with pytest.raises(ValueError, match="label_col"):
        LabeledDataset.from_csv(tmp_csv, text_col="srcs", label_col="nonexistent")


def test_labeled_dataset_len(tmp_csv):
    ds = LabeledDataset.from_csv(tmp_csv, text_col="srcs", label_col="label")
    assert len(ds) == 10


def test_two_group_dataset_len(tmp_csv):
    ds = TwoGroupDataset.from_csv_pair(tmp_csv, tmp_csv, text_col="srcs")
    assert len(ds) == 20
