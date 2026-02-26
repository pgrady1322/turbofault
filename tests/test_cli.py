"""
TurboFault v0.1.0

test_cli.py — Tests for the Click CLI interface.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import pytest
from click.testing import CliRunner

from turbofault.cli import main, ALL_MODELS, TABULAR_MODELS, DEEP_MODELS


class TestCLISetup:
    """Test CLI group and basic configuration."""

    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "TurboFault" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_model_sets(self):
        assert "xgboost" in TABULAR_MODELS
        assert "random_forest" in TABULAR_MODELS
        assert "ridge" in TABULAR_MODELS
        assert "lstm" in DEEP_MODELS
        assert "gru" in DEEP_MODELS
        assert "transformer" in DEEP_MODELS
        assert "cnn1d" in DEEP_MODELS
        assert ALL_MODELS == TABULAR_MODELS | DEEP_MODELS


class TestDownloadCommand:
    def test_download_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output


class TestTrainCommand:
    def test_train_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--subset" in result.output
        assert "--config" in result.output
        assert "--epochs" in result.output

    def test_train_requires_model(self):
        runner = CliRunner()
        result = runner.invoke(main, ["train"])
        assert result.exit_code != 0


class TestTuneCommand:
    def test_tune_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["tune", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--n-trials" in result.output


class TestEvaluateCommand:
    def test_evaluate_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--subset" in result.output
        assert "--model-path" in result.output
        assert "--output" in result.output

    def test_evaluate_requires_model(self):
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate"])
        assert result.exit_code != 0


class TestExplainCommand:
    def test_explain_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["explain", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--method" in result.output
        assert "--top-n" in result.output

    def test_explain_requires_model(self):
        runner = CliRunner()
        result = runner.invoke(main, ["explain"])
        assert result.exit_code != 0

    def test_explain_rejects_deep_models(self):
        runner = CliRunner()
        # Deep models not supported for explain — exit code 1
        result = runner.invoke(main, ["explain", "-m", "lstm",
                                      "-d", "/nonexistent/path"])
        assert result.exit_code != 0

# TurboFault v0.1.0
# Any usage is subject to this software's license.
