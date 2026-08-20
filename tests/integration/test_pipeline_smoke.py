import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def gpt2_run(tmp_path_factory):
    """Run cap capture + stats on gpt2 with 5 samples per group. CPU only."""
    import importlib.resources as res

    from cap.data.loaders import LabeledDataset
    from cap.data.prompt_templates import FACTUALITY
    from cap.experiments.capture import CaptureExperiment

    csv_path = res.files("cap.data.files") / "english_smol_with_gen.csv"

    ds = LabeledDataset.from_csv(csv_path, text_col="srcs", label_col="label", n_samples=5, seed=42)
    a, b = ds.as_groups()

    prompted_a = [FACTUALITY.apply(t) for t in a]
    prompted_b = [FACTUALITY.apply(t) for t in b]

    out = tmp_path_factory.mktemp("smoke") / "run"
    exp = CaptureExperiment(model_path="gpt2", device="cpu", trust_remote_code=False)
    exp.run_contrast(
        prompts1=prompted_a,
        prompts2=prompted_b,
        label1="faithful",
        label2="unfaithful",
        output_path=out,
    )
    return out


def test_h5_files_exist(gpt2_run):
    assert (gpt2_run / "activations.h5").exists()
    assert (gpt2_run / "statistics.h5").exists()


def test_manifest_is_valid(gpt2_run):
    import cap
    from cap.cli._manifest import read_manifest, write_manifest

    write_manifest(
        gpt2_run,
        cap_version=cap.__version__,
        command=["test"],
        model="gpt2",
        data_spec={"path": "test.csv"},
        prompt_template_dict={"type": "factuality", "template": "Is {text} true?"},
        seed=42,
        device="cpu",
    )
    m = read_manifest(gpt2_run)
    assert m["model"] == "gpt2"


def test_statistics_have_expected_layers(gpt2_run):
    from cap.utils.h5_utils import H5Store

    stats = H5Store.load_statistics(gpt2_run / "statistics.h5")
    assert len(stats) > 0
    for _layer_name, layer_stats in stats.items():
        assert "t_stats" in layer_stats
        assert "significant" in layer_stats
