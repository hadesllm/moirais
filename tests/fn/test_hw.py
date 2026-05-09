"""Tests for moirais.fn.hw -- Hardy-Weinberg equilibrium test."""

import pytest
from moirais.fn.hw import hardy_weinberg_test


class TestHardyWeinberg:
    def test_equilibrium_not_rejected(self):
        """Counts consistent with HWE should yield p > 0.05."""
        res = hardy_weinberg_test(n_AA=36, n_Aa=48, n_aa=16)
        assert res.name == "Hardy-Weinberg"
        assert res.p_value > 0.05
        assert res.n == 100

    def test_disequilibrium_rejected(self):
        """Extreme counts should reject HWE (p < 0.05)."""
        res = hardy_weinberg_test(n_AA=90, n_Aa=0, n_aa=10)
        assert res.p_value < 0.05

    def test_allele_freq_correct(self):
        """Check allele frequency calculation."""
        res = hardy_weinberg_test(n_AA=25, n_Aa=50, n_aa=25)
        assert res.extra["p_allele"] == pytest.approx(0.5, abs=1e-10)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            hardy_weinberg_test(n_AA=-1, n_Aa=10, n_aa=10)
