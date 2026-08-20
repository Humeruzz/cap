from pathlib import Path

import pytest
import typer

from cap.cli._options import parse_experiment_paths, parse_resolutions


class TestParseResolutions:
    def test_comma_separated(self):
        assert parse_resolutions("32,64,128") == [32, 64, 128]

    def test_space_separated_single_token(self):
        assert parse_resolutions("32 64 128") == [32, 64, 128]

    def test_single_value(self):
        assert parse_resolutions("64") == [64]

    def test_non_integer_is_rejected(self):
        with pytest.raises(typer.BadParameter):
            parse_resolutions("32,abc")

    def test_empty_is_rejected(self):
        with pytest.raises(typer.BadParameter):
            parse_resolutions("   ")


class TestParseExperimentPaths:
    def test_repeated_flag_form(self):
        assert parse_experiment_paths(["runs/en", "runs/zh"]) == [
            Path("runs/en"),
            Path("runs/zh"),
        ]

    def test_comma_separated_form(self):
        assert parse_experiment_paths(["runs/en,runs/zh"]) == [
            Path("runs/en"),
            Path("runs/zh"),
        ]

    def test_mixed_forms(self):
        assert parse_experiment_paths(["runs/en,runs/zh", "runs/de"]) == [
            Path("runs/en"),
            Path("runs/zh"),
            Path("runs/de"),
        ]

    def test_paths_with_spaces_are_preserved(self):
        # Unlike --downsample, whitespace must not split: spaces are legal in paths.
        assert parse_experiment_paths(["my runs/en"]) == [Path("my runs/en")]

    def test_surrounding_whitespace_is_trimmed(self):
        assert parse_experiment_paths(["runs/en , runs/zh"]) == [
            Path("runs/en"),
            Path("runs/zh"),
        ]

    def test_empty_segments_are_dropped(self):
        assert parse_experiment_paths(["runs/en,,runs/zh"]) == [
            Path("runs/en"),
            Path("runs/zh"),
        ]

    def test_none_and_empty(self):
        assert parse_experiment_paths(None) == []
        assert parse_experiment_paths([]) == []
