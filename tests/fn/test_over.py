"""Tests for moirais.fn.over -- Overlap diagnostics."""

import numpy as np
import pytest
from moirais.fn.over import overlap_diagnostics


class TestOverlapDiagnostics:
    def test_identical_distributions(self):
        rng = np.random.default_rng(42)
        ps = rng.uniform(0.2, 0.8, 200)
        result = overlap_diagnostics(ps, ps)
        assert result["pct_trimmed"] == pytest.approx(0.0)

    def test_no_overlap(self):
        pt = np.array([0.8, 0.85, 0.9, 0.95])
        pc = np.array([0.05, 0.1, 0.15, 0.2])
        result = overlap_diagnostics(pt, pc)
        # Overlap range should be narrow or empty-ish
        lo, hi = result["overlap_range"]
        assert hi >= lo  # valid range

    def test_ks_stat_range(self):
        rng = np.random.default_rng(42)
        pt = rng.uniform(0.3, 0.7, 100)
        pc = rng.uniform(0.2, 0.8, 100)
        result = overlap_diagnostics(pt, pc)
        assert 0 <= result["ks_stat"] <= 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            overlap_diagnostics([], [0.5])
