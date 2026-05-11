"""Tests for morie.fn.sxmrg -- Sex-stratified meta-regression."""

import numpy as np
import pytest
from morie.fn.sxmrg import sxmrg


class TestSxmrg:
    def test_equal_effects_no_diff(self):
        bm = np.array([0.5, 0.3, -0.2])
        sm = np.array([0.1, 0.1, 0.1])
        bf = np.array([0.5, 0.3, -0.2])
        sf = np.array([0.1, 0.1, 0.1])
        res = sxmrg(bm, sm, bf, sf)
        diff = np.array(res.extra["beta_diff"])
        np.testing.assert_allclose(diff, 0.0, atol=1e-10)

    def test_different_effects(self):
        bm = np.array([1.0, 0.0])
        sm = np.array([0.1, 0.1])
        bf = np.array([0.0, 0.0])
        sf = np.array([0.1, 0.1])
        res = sxmrg(bm, sm, bf, sf)
        p = np.array(res.extra["p_diff"])
        assert p[0] < 0.05

    def test_combined_estimates(self):
        bm = np.array([0.4])
        sm = np.array([0.1])
        bf = np.array([0.6])
        sf = np.array([0.1])
        res = sxmrg(bm, sm, bf, sf)
        combined = res.extra["beta_combined"][0]
        assert 0.4 < combined < 0.6

    def test_negative_se_raises(self):
        with pytest.raises(ValueError):
            sxmrg(np.array([0.5]), np.array([-0.1]),
                   np.array([0.5]), np.array([0.1]))

    def test_mismatched_length(self):
        with pytest.raises(ValueError):
            sxmrg(np.array([0.5, 0.3]), np.array([0.1]),
                   np.array([0.5]), np.array([0.1]))
