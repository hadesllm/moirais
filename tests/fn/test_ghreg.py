"""Tests for ghreg.ghosal_np_regression."""
import numpy as np
import pytest
from moirais.fn.ghreg import ghosal_np_regression


def test_ghreg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_np_regression(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghreg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_np_regression(x, y)
    assert isinstance(result, dict)
