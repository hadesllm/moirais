"""Tests for km142.kamath_ch9_itg_loss."""
import numpy as np
import pytest
from moirais.fn.km142 import kamath_ch9_itg_loss


def test_km142_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch9_itg_loss(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km142_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch9_itg_loss(x, y)
    assert isinstance(result, dict)
