"""Tests for morie.fn.bonf -- Bonferroni correction."""

import pytest
from morie.fn.bonf import bonferroni_correction


class TestBonferroni:
    def test_known_adjustment(self):
        """[0.01, 0.04, 0.06] * 3 => [0.03, 0.12, 0.18]."""
        res = bonferroni_correction([0.01, 0.04, 0.06])
        adj = res.extra["adjusted"]
        assert adj[0] == pytest.approx(0.03, abs=1e-10)
        assert adj[1] == pytest.approx(0.12, abs=1e-10)
        assert adj[2] == pytest.approx(0.18, abs=1e-10)

    def test_capped_at_one(self):
        """Adjusted p-values should never exceed 1.0."""
        res = bonferroni_correction([0.5, 0.6])
        for p in res.extra["adjusted"]:
            assert p <= 1.0

    def test_significance_count(self):
        """Only p-values < alpha after adjustment are significant."""
        res = bonferroni_correction([0.01, 0.04, 0.06], alpha=0.05)
        assert res.value == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bonferroni_correction([])
