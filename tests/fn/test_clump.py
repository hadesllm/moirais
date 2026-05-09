"""Tests for moirais.fn.clump — LD clumping."""
import numpy as np
import pytest
from moirais.fn.clump import ld_clumping


class TestLDClumping:
    def test_basic(self):
        pv = np.array([1e-9, 0.5, 0.3, 1e-10, 0.8])
        ld = np.eye(5)
        ld[0, 3] = 0.5
        ld[3, 0] = 0.5
        res = ld_clumping(pv, ld, threshold=5e-8, r2_threshold=0.1)
        assert res.value >= 1

    def test_no_significant(self):
        pv = np.array([0.1, 0.5, 0.9])
        ld = np.eye(3)
        res = ld_clumping(pv, ld, threshold=5e-8)
        assert res.value == 0
