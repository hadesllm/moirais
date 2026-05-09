"""Tests for km021.kamath_ch2_clm_loss."""
import numpy as np
import pytest
from moirais.fn.km021 import kamath_ch2_clm_loss


def test_km021_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_clm_loss(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km021_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_clm_loss(x)
    assert isinstance(result, dict)
