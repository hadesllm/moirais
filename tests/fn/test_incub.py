"""Tests for moirais.fn.incub -- Incubation period distribution."""

import pytest
import numpy as np
from moirais.fn.incub import incubation_period


class TestIncubation:
    def test_lognormal(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(mean=1.5, sigma=0.5, size=200)
        res = incubation_period(data, distribution="lognormal")
        assert res.measure == "incubation_period"
        assert res.estimate > 0

    def test_gamma(self):
        rng = np.random.default_rng(42)
        data = rng.gamma(shape=5, scale=1.0, size=100)
        res = incubation_period(data, distribution="gamma")
        assert res.estimate > 0

    def test_too_few(self):
        with pytest.raises(ValueError):
            incubation_period([1.0, 2.0])
