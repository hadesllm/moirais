"""Tests for morie.progress — pipeline progress tracking."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from morie.progress import ModuleResult, PipelineTracker, execute_pipeline_with_progress


# ---------------------------------------------------------------------------
# ModuleResult
# ---------------------------------------------------------------------------


class TestModuleResult:
    def test_defaults(self):
        r = ModuleResult(name="test-module")
        assert r.name == "test-module"
        assert r.status == "pending"
        assert r.elapsed_seconds == 0.0
        assert r.output_files_expected == 0
        assert r.output_files_actual == 0
        assert r.error_message is None
        assert r.outputs == {}

    def test_success_state(self):
        r = ModuleResult(
            name="m",
            status="success",
            elapsed_seconds=1.5,
            output_files_expected=3,
            output_files_actual=3,
        )
        assert r.status == "success"
        assert r.elapsed_seconds == 1.5


# ---------------------------------------------------------------------------
# PipelineTracker — plain (non-TTY) mode
# ---------------------------------------------------------------------------


class TestPipelineTrackerPlain:
    @patch("morie.progress.run_module")
    def test_run_all_success(self, mock_run):
        mock_run.return_value = {"table1": MagicMock(), "table2": MagicMock()}

        tracker = PipelineTracker(
            ["power-design"],
            cpads_csv="fake.csv",
            use_live=False,
            track_carbon=False,
        )
        results = tracker.run()

        assert len(results) == 1
        assert results[0].status == "success"
        assert results[0].output_files_actual == 2
        assert results[0].elapsed_seconds >= 0
        mock_run.assert_called_once_with(
            "power-design", cpads_csv="fake.csv", output_dir=None
        )

    @patch("morie.progress.run_module", side_effect=RuntimeError("bad data"))
    def test_run_handles_error(self, mock_run):
        tracker = PipelineTracker(
            ["power-design"],
            cpads_csv="fake.csv",
            use_live=False,
            track_carbon=False,
        )
        results = tracker.run()

        assert len(results) == 1
        assert results[0].status == "error"
        assert "bad data" in results[0].error_message

    @patch("morie.progress.run_module")
    def test_run_multiple_modules(self, mock_run):
        mock_run.return_value = {"out": MagicMock()}

        tracker = PipelineTracker(
            ["power-design", "logistic-models"],
            cpads_csv="fake.csv",
            use_live=False,
            track_carbon=False,
        )
        results = tracker.run()

        assert len(results) == 2
        assert all(r.status == "success" for r in results)
        assert mock_run.call_count == 2

    @patch("morie.progress.run_module")
    def test_run_single(self, mock_run):
        mock_run.return_value = {"t": MagicMock()}

        tracker = PipelineTracker(
            [],
            cpads_csv="fake.csv",
            use_live=False,
            track_carbon=False,
        )
        result = tracker.run_single("power-design")

        assert result.status == "success"
        assert result.output_files_actual == 1

    @patch("morie.progress.run_module")
    def test_output_dir_passed_through(self, mock_run):
        mock_run.return_value = {}

        tracker = PipelineTracker(
            ["power-design"],
            cpads_csv="fake.csv",
            output_dir="/tmp/out",
            use_live=False,
            track_carbon=False,
        )
        tracker.run()

        mock_run.assert_called_once_with(
            "power-design", cpads_csv="fake.csv", output_dir="/tmp/out"
        )


# ---------------------------------------------------------------------------
# PipelineTracker — summary table
# ---------------------------------------------------------------------------


class TestSummaryTable:
    @patch("morie.progress.run_module")
    def test_summary_table_builds(self, mock_run):
        mock_run.return_value = {"out": MagicMock()}

        tracker = PipelineTracker(
            ["power-design"],
            cpads_csv="fake.csv",
            use_live=False,
            track_carbon=False,
        )
        tracker.run()

        table = tracker.summary_table()
        assert table.title == "Pipeline Summary"
        assert len(table.columns) == 4
        assert table.row_count == 1


# ---------------------------------------------------------------------------
# execute_pipeline_with_progress
# ---------------------------------------------------------------------------


class TestExecutePipelineWithProgress:
    @patch("morie.progress.run_module")
    def test_returns_zero_on_success(self, mock_run, monkeypatch):
        mock_run.return_value = {"out": MagicMock()}
        # Force non-TTY to avoid rich Live display
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        code = execute_pipeline_with_progress(
            modules=["power-design"],
            cpads_csv="fake.csv",
            silent=True,
            track_carbon=False,
        )
        assert code == 0

    @patch("morie.progress.run_module", side_effect=RuntimeError("fail"))
    def test_returns_one_on_failure(self, mock_run, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        code = execute_pipeline_with_progress(
            modules=["power-design"],
            cpads_csv="fake.csv",
            silent=True,
            track_carbon=False,
        )
        assert code == 1

    @patch("morie.progress.run_module")
    def test_abort_on_no_confirm(self, mock_run, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")

        code = execute_pipeline_with_progress(
            modules=["power-design"],
            cpads_csv="fake.csv",
            silent=False,
            track_carbon=False,
        )
        assert code == 1
        mock_run.assert_not_called()
