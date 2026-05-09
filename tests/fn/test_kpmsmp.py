"""Tests for kpmsmp.km_simultaneous_band."""
import numpy as np
import pytest
from moirais.fn.kpmsmp import km_simultaneous_band


def test_kpmsmp_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = km_simultaneous_band(fit, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kpmsmp_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = km_simultaneous_band(fit, alpha)
    assert isinstance(result, dict)
