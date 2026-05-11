"""Tests for morie.fn.bp_ts — Bai-Perron structural break test."""

import numpy as np
import pytest

from morie.fn.bp_ts import bp_ts


class TestBpTs:
    """Tests for bp_ts()."""

    def test_detects_single_break(self):
        """Detects a single mean shift in a piecewise-constant series."""
        rng = np.random.default_rng(42)
        y = np.concatenate([
            rng.normal(0, 0.5, 60),
            rng.normal(5, 0.5, 60),
        ])
        result = bp_ts(y, max_breaks=3)
        assert result["n_breaks"] >= 1
        assert len(result["bic"]) >= 2

    def test_no_break_constant(self):
        """Constant series (no noise) produces zero breaks."""
        y = np.full(100, 5.0)
        result = bp_ts(y, max_breaks=2)
        assert result["n_breaks"] == 0

    def test_segments_cover_all_data(self):
        """Segments cover the entire series."""
        rng = np.random.default_rng(99)
        y = np.concatenate([rng.normal(0, 1, 40), rng.normal(3, 1, 40)])
        result = bp_ts(y, max_breaks=2)
        segs = result["segments"]
        assert segs[0][0] == 0
        assert segs[-1][1] == len(y)
