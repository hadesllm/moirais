"""Tests for morie.fn.snphw -- SNP Hardy-Weinberg exact test."""

import numpy as np
import pytest
from morie.fn.snphw import snphw


class TestSnphw:
    def test_hwe_equilibrium(self):
        res = snphw([25, 50, 25])
        assert res.p_value > 0.05

    def test_hwe_deviation(self):
        res = snphw([50, 0, 50])
        assert res.p_value < 0.05

    def test_all_homozygous(self):
        res = snphw([100, 0, 0])
        assert res.p_value == pytest.approx(1.0)

    def test_multi_snp(self):
        counts = np.array([[25, 50, 25], [50, 0, 50]])
        res = snphw(counts)
        assert len(res.extra["p_values"]) == 2

    def test_negative_counts_raise(self):
        with pytest.raises(ValueError):
            snphw([-1, 50, 25])

    def test_p_value_bounded(self):
        res = snphw([30, 40, 30])
        assert 0 <= res.p_value <= 1
