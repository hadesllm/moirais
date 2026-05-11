"""Tests for bifurcation_data."""
import pytest
from morie.fn.bifrc import bifurcation_data

class TestBifurcation:
    def test_basic(self):
        r = bifurcation_data(n_r=50, n_iter=100, n_last=20)
        assert r.extra["n_points"] == 50 * 20

    def test_feigenbaum(self):
        r = bifurcation_data()
        assert r.extra["feigenbaum_r1"] == pytest.approx(3.0)
