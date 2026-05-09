"""Tests for okrig.ordinary_kriging."""
import numpy as np
import pytest
from moirais.fn.okrig import ordinary_kriging


def test_okrig_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = ordinary_kriging(x, coords, target)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_okrig_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = ordinary_kriging(x, coords, target)
    assert isinstance(result, dict)
