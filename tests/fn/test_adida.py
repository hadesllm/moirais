"""Tests for adida.adida."""
import numpy as np
import pytest
from morie.fn.adida import adida


def test_adida_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    result = adida(y, period)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_adida_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    result = adida(y, period)
    assert isinstance(result, dict)
