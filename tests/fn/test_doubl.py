"""Tests for moirais.fn.doubl — Doubling time."""

import numpy as np
import pytest

from moirais.fn.doubl import doubling_time


class TestDoublingTime:
    def test_from_rate(self):
        res = doubling_time(r=0.1)
        assert res.estimate == pytest.approx(np.log(2) / 0.1, rel=1e-6)

    def test_from_incidence(self):
        inc = 10 * np.exp(0.05 * np.arange(30))
        res = doubling_time(incidence=inc)
        assert 10 < res.estimate < 20

    def test_negative_rate(self):
        with pytest.raises(ValueError):
            doubling_time(r=-0.1)
