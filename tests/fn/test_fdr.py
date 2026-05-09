"""Tests for moirais.fn.fdr -- Benjamini-Hochberg FDR."""

import pytest
from moirais.fn.fdr import benjamini_hochberg


class TestBenjaminiHochberg:
    def test_known_significance(self):
        """[0.001, 0.01, 0.03, 0.5] at alpha=0.05 => first 3 significant."""
        res = benjamini_hochberg([0.001, 0.01, 0.03, 0.5], alpha=0.05)
        assert res.name == "Benjamini-Hochberg"
        assert res.value == 3
        sig = res.extra["significant"]
        assert sig[:3] == [True, True, True]
        assert sig[3] is False

    def test_adjusted_monotonic(self):
        """Adjusted p-values (sorted) should be non-decreasing."""
        pv = [0.005, 0.01, 0.02, 0.04, 0.10, 0.80]
        res = benjamini_hochberg(pv)
        adj = res.extra["adjusted"]
        sorted_adj = sorted(adj)
        for a, b in zip(sorted_adj, sorted_adj[1:]):
            assert a <= b + 1e-12

    def test_all_significant(self):
        """Very small p-values should all be significant."""
        res = benjamini_hochberg([0.001, 0.002, 0.003])
        assert res.value == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            benjamini_hochberg([])
