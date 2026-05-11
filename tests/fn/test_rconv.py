"""Test convergence_rate (rconv)."""
import numpy as np
import pytest

from morie.fn.rconv import convergence_rate, rconv
from morie.fn._containers import DescriptiveResult


class TestConvergenceRate:
    def test_exponential_decay(self):
        errors = np.exp(-0.5 * np.arange(20))
        result = convergence_rate(errors)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "convergence_rate"
        assert np.isclose(result.value, 0.5, atol=0.01)

    def test_converging_flag(self):
        errors = np.exp(-0.3 * np.arange(10))
        result = convergence_rate(errors)
        assert result.extra["converging"] is True

    def test_too_few(self):
        with pytest.raises(ValueError):
            convergence_rate([1.0])

    def test_alias(self):
        assert rconv is convergence_rate
