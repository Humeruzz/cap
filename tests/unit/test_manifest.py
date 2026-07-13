import pytest
from cap.cli._manifest import write_manifest, read_manifest


def test_write_and_read_round_trip(tmp_path):
    write_manifest(
        tmp_path,
        cap_version="0.2.0",
        command=["cap", "capture"],
        model="gpt2",
        data_spec={
            "path": "data.csv",
            "text_column": "srcs",
            "label_column": "label",
            "n_samples": 0,
        },
        prompt_template_dict={"type": "factuality", "template": "Is {text} true?"},
        seed=42,
        device="cpu",
    )
    m = read_manifest(tmp_path)
    assert m["model"] == "gpt2"
    assert m["seed"] == 42


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path)
