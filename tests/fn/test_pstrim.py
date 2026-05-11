"""Tests for propensity_trim."""
import numpy as np, pytest
from morie.fn.pstrim import propensity_trim

class TestPSTrim:
    def test_basic(self):
        ps = np.linspace(0, 1, 100)
        r = propensity_trim(ps, trim=0.05)
        assert r.extra["n_trimmed"] > 0

    def test_no_trim(self):
        ps = np.linspace(0.2, 0.8, 50)
        r = propensity_trim(ps, trim=0.0)
        assert r.value == 50
