"""Tests for lyexp -- Lyapunov exponent estimation."""
import numpy as np
import pytest
from moirais.fn.lyexp import lyexp
from moirais.fn._containers import DescriptiveResult


def test_lyexp_basic():
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.standard_normal(500))
    result = lyexp(x, m=3, tau=1)
    assert isinstance(result, DescriptiveResult)
    assert "lyapunov_exponent" in result.extra


def test_lyexp_chaotic():
    x = np.zeros(2000)
    x[0] = 0.1
    for i in range(1, 2000):
        x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])
    result = lyexp(x, m=3, tau=1)
    assert result.value > 0


def test_lyexp_too_short():
    with pytest.raises(ValueError):
        lyexp(np.array([1.0, 2.0, 3.0]), m=5, tau=2)
