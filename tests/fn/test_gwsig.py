"""Tests for moirais.fn.gwsig -- Genome-wide significance."""

import numpy as np
import pytest
from moirais.fn.gwsig import gwsig


class TestGwsig:
    def test_bonferroni(self):
        pvals = np.array([1e-8, 0.01, 0.05, 0.5, 0.9])
        res = gwsig(pvals, method="bonferroni")
        assert res.extra["threshold"] == pytest.approx(0.01, rel=1e-6)
        assert 0 in res.extra["significant_indices"]

    def test_bh_fdr(self):
        rng = np.random.default_rng(42)
        pvals = np.concatenate([np.array([1e-6, 1e-5]), rng.uniform(0.1, 1, 98)])
        res = gwsig(pvals, method="bh")
        assert res.statistic >= 2

    def test_sidak(self):
        pvals = np.array([1e-8, 0.5])
        res = gwsig(pvals, method="sidak")
        assert 0 in res.extra["significant_indices"]

    def test_adjusted_bounded(self):
        pvals = np.array([0.01, 0.05, 0.1])
        res = gwsig(pvals, method="bonferroni")
        adj = np.array(res.extra["adjusted_pvalues"])
        assert np.all(adj <= 1.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            gwsig(np.ones(10), method="unknown")

    def test_empty(self):
        res = gwsig(np.array([]))
        assert res.statistic == 0.0
