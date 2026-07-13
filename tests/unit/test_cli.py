from typer.testing import CliRunner
from cap.cli.main import app

runner = CliRunner()


def test_cap_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "cap" in result.output.lower()


def test_capture_help():
    result = runner.invoke(app, ["capture", "--help"])
    assert result.exit_code == 0


def test_stats_help():
    result = runner.invoke(app, ["stats", "--help"])
    assert result.exit_code == 0


def test_patch_help():
    result = runner.invoke(app, ["patch", "--help"])
    assert result.exit_code == 0


def test_visualize_help():
    result = runner.invoke(app, ["visualize", "--help"])
    assert result.exit_code == 0


def test_run_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_capture_missing_required_flags():
    result = runner.invoke(app, ["capture"])
    assert result.exit_code != 0


def test_invalid_type_flag():
    result = runner.invoke(
        app,
        [
            "capture",
            "--model",
            "gpt2",
            "--output",
            "/tmp/x",
            "--data",
            "x.csv",
            "--type",
            "nonexistent_type",
        ],
    )
    assert result.exit_code != 0
