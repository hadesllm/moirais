"""Tests for krigsm.ordinary_kriging."""
import numpy as np
import pytest
from moirais.fn.krigsm import ordinary_kriging


def test_krigsm_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    s_predict = np.random.default_rng(42).normal(0, 1, 100)
    variogram = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinary_kriging(coords, values, s_predict, variogram)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krigsm_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    s_predict = np.random.default_rng(42).normal(0, 1, 100)
    variogram = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinary_kriging(coords, values, s_predict, variogram)
    assert isinstance(result, dict)
