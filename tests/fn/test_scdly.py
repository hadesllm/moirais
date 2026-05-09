"""Tests for moirais.fn.scdly -- Surveillance case delay."""

import pytest
import numpy as np
from moirais.fn.scdly import case_delay


class TestCaseDelay:
    def test_gamma(self):
        rng = np.random.default_rng(42)
        data = rng.gamma(shape=2, scale=3, size=100)
        res = case_delay(data, distribution="gamma")
        assert res.measure == "case_delay"
        assert res.estimate > 0

    def test_exponential(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=5, size=100)
        res = case_delay(data, distribution="exponential")
        assert res.estimate > 0

    def test_too_few(self):
        with pytest.raises(ValueError):
            case_delay([1.0, 2.0])
