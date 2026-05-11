"""Tests for morie.fn.gnblk -- Genomic block bootstrap."""

import numpy as np
import pytest
from morie.fn.gnblk import gnblk


class TestGnblk:
    def test_basic_bootstrap(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(20, 100)).astype(float)
        res = gnblk(lambda x: np.mean(x), Z, block_size=10, n_bootstrap=100)
        assert res.extra["se"] > 0

    def test_ci_contains_statistic(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(20, 100)).astype(float)
        res = gnblk(lambda x: np.mean(x), Z, block_size=10, n_bootstrap=500)
        assert res.extra["ci_lower"] <= res.statistic <= res.extra["ci_upper"]

    def test_large_block(self):
        Z = np.ones((5, 10))
        res = gnblk(lambda x: np.mean(x), Z, block_size=20, n_bootstrap=50)
        assert res.extra["se"] == pytest.approx(0.0, abs=1e-10)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            gnblk(lambda x: 0, np.ones((5, 10)), block_size=0)
