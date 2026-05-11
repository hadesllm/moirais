"""Tests for morie.fn.theta — Watterson's theta."""
import pytest
from morie.fn.theta import watterson_theta


class TestWattersonTheta:
    def test_basic(self):
        res = watterson_theta(S=10, n=20)
        assert res.statistic > 0
        assert res.n == 20

    def test_zero_sites(self):
        res = watterson_theta(S=0, n=10)
        assert res.statistic == 0.0

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            watterson_theta(S=5, n=1)
