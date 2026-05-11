"""Tests for morie.fn.fstfr -- Forensic Fst/theta."""

import numpy as np
import pytest
from morie.fn.fstfr import fstfr


class TestFstfr:
    def test_identical_pops_zero(self):
        freqs = np.array([[0.3, 0.5], [0.3, 0.5]])
        res = fstfr(freqs)
        assert res.statistic == pytest.approx(0.0, abs=1e-6)

    def test_divergent_pops(self):
        freqs = np.array([[0.9, 0.1], [0.1, 0.9]])
        res = fstfr(freqs)
        assert res.statistic > 0.3

    def test_bounded_zero_one(self):
        rng = np.random.default_rng(42)
        freqs = rng.uniform(0.05, 0.95, size=(4, 10))
        res = fstfr(freqs)
        assert 0 <= res.statistic <= 1

    def test_too_few_pops(self):
        with pytest.raises(ValueError):
            fstfr(np.array([[0.5, 0.5]]))
