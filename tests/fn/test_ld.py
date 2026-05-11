"""Tests for morie.fn.ld -- Linkage disequilibrium."""

import numpy as np
import pytest
from morie.fn.ld import linkage_disequilibrium


class TestLD:
    def test_perfectly_linked(self):
        """Identical allele arrays => r^2 = 1."""
        a = [1, 0, 1, 0, 1, 0, 1, 0]
        res = linkage_disequilibrium(a, a)
        assert res.name == "LD_r2"
        assert res.statistic == pytest.approx(1.0, abs=1e-10)

    def test_independent(self):
        """Large random independent loci => r^2 near 0."""
        rng = np.random.default_rng(42)
        a = rng.integers(0, 2, size=10000)
        b = rng.integers(0, 2, size=10000)
        res = linkage_disequilibrium(a, b)
        assert res.statistic < 0.01

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            linkage_disequilibrium([0, 1], [0, 1, 0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            linkage_disequilibrium([], [])
