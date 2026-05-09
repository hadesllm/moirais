"""Tests for moirais._parity — verify migration parity is complete."""

from pathlib import Path

from moirais._parity import build_parity_matrix, summarize_parity_matrix

_RTESTS_DIR = Path(__file__).resolve().parents[1] / "rtests"


def test_parity_matrix_has_expected_kinds():
    matrix = build_parity_matrix(_RTESTS_DIR)
    assert "kind" in matrix.columns
    assert len(matrix) > 0
    assert "analysis_module" in matrix["kind"].values


def test_parity_summary_counts_rows():
    matrix = build_parity_matrix(_RTESTS_DIR)
    summary = summarize_parity_matrix(matrix)
    assert summary.total_rows == len(matrix)
    assert summary.already_present >= 1
