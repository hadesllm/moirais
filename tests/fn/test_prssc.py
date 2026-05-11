"""Tests for morie.fn.prssc -- Polygenic risk score."""

import numpy as np
import pytest
from morie.fn.prssc import prssc


class TestPrssc:
    def test_basic_prs(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(20, 10)).astype(float)
        betas = rng.standard_normal(10) * 0.1
        res = prssc(Z, betas)
        assert len(res.extra["scores"]) == 20

    def test_threshold_filters(self):
        Z = np.ones((5, 10))
        betas = np.ones(10)
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5,
                          0.6, 0.7, 0.8, 0.9, 1.0])
        res = prssc(Z, betas, p_values=pvals, p_threshold=0.05)
        assert res.extra["n_snps_used"] == 2

    def test_all_snps_default(self):
        Z = np.ones((5, 10))
        betas = np.ones(10)
        res = prssc(Z, betas)
        assert res.extra["n_snps_used"] == 10

    def test_standardized(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(20, 10)).astype(float)
        betas = np.ones(10)
        res = prssc(Z, betas, standardize=True)
        assert abs(np.mean(res.extra["scores"])) < 1.0

    def test_mismatched_betas(self):
        with pytest.raises(ValueError):
            prssc(np.ones((5, 10)), np.ones(5))
