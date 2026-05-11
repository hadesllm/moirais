"""Tests for morie.fn.bfmi -- BFMI diagnostic."""

import numpy as np
from morie.fn.bfmi import bayesian_fmi


def test_returns_dict():
    result = bayesian_fmi([1.0, 2.0, 1.5, 2.5, 1.8])
    assert isinstance(result, dict)
    assert "bfmi" in result


def test_good_mixing():
    rng = np.random.default_rng(42)
    energy = rng.normal(0, 1, 1000)
    result = bayesian_fmi(energy)
    assert result["bfmi"] > 0.3
    assert result["adequate"] is True


def test_poor_mixing():
    energy = np.cumsum(np.ones(100))
    result = bayesian_fmi(energy)
    assert result["adequate"] is False


def test_constant_energy():
    result = bayesian_fmi(np.ones(100))
    assert result["bfmi"] == 1.0


def test_too_short():
    try:
        bayesian_fmi([1.0, 2.0])
        assert False
    except ValueError:
        pass
