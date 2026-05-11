"""Tests for corrn -- Correlation dimension."""
import numpy as np
import pytest
from morie.fn.corrn import corrn
from morie.fn._containers import DescriptiveResult


def test_corrn_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(300)
    result = corrn(x, m=3, tau=1)
    assert isinstance(result, DescriptiveResult)
    assert "d2" in result.extra


def test_corrn_positive():
    x = np.zeros(1000)
    x[0] = 0.1
    for i in range(1, 1000):
        x[i] = 3.8 * x[i - 1] * (1 - x[i - 1])
    result = corrn(x, m=3, tau=1)
    assert result.value > 0


def test_corrn_too_short():
    with pytest.raises(ValueError):
        corrn(np.array([1.0, 2.0]), m=5, tau=2)
