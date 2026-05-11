"""Tests for morie.fn.rmpob -- Random match probability."""

import numpy as np
import pytest
from morie.fn.rmpob import rmpob


class TestRmpob:
    def test_single_locus_homozygote(self):
        res = rmpob([(0.1, 0.1)])
        assert res.p_value == pytest.approx(0.01, rel=1e-6)

    def test_single_locus_heterozygote(self):
        res = rmpob([(0.3, 0.2)])
        assert res.p_value == pytest.approx(0.12, rel=1e-6)

    def test_multi_locus_product(self):
        res = rmpob([(0.1, 0.1), (0.2, 0.2)])
        expected = 0.01 * 0.04
        assert res.p_value == pytest.approx(expected, rel=1e-6)

    def test_theta_correction_increases_rmp(self):
        res_no = rmpob([(0.1, 0.1)])
        res_th = rmpob([(0.1, 0.1)], theta=0.03)
        assert res_th.p_value > res_no.p_value

    def test_log10_statistic(self):
        res = rmpob([(0.1, 0.1)])
        assert res.statistic == pytest.approx(-np.log10(0.01), rel=1e-6)

    def test_invalid_freq(self):
        with pytest.raises(ValueError):
            rmpob([(-0.1, 0.5)])
