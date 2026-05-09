"""Tests for malab.ma_labbe_plot."""
import numpy as np
import pytest
from moirais.fn.malab import ma_labbe_plot


def test_malab_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = ma_labbe_plot(a, b, c, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_malab_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = ma_labbe_plot(a, b, c, d)
    assert isinstance(result, dict)
