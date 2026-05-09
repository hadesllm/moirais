"""Tests for moirais.fn.bhpd -- HPD interval."""

import numpy as np
from moirais.fn.bhpd import hpd_interval


def test_returns_dict():
    result = hpd_interval([1, 2, 3, 4, 5])
    assert isinstance(result, dict)
    assert "hpd_lower" in result
    assert "hpd_upper" in result


def test_symmetric_distribution():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 10000)
    result = hpd_interval(samples, prob=0.95)
    assert abs(result["hpd_lower"] + result["hpd_upper"]) < 0.3


def test_hpd_shorter_than_equal_tailed():
    rng = np.random.default_rng(42)
    samples = rng.lognormal(0, 1, 10000)
    hpd = hpd_interval(samples, prob=0.95)
    hpd_width = hpd["width"]
    q_lo = np.percentile(samples, 2.5)
    q_hi = np.percentile(samples, 97.5)
    et_width = q_hi - q_lo
    assert hpd_width <= et_width + 0.1


def test_width_positive():
    rng = np.random.default_rng(42)
    result = hpd_interval(rng.normal(0, 1, 100))
    assert result["width"] > 0


def test_too_short():
    try:
        hpd_interval([1])
        assert False
    except ValueError:
        pass


def test_invalid_prob():
    try:
        hpd_interval([1, 2, 3], prob=0)
        assert False
    except ValueError:
        pass
