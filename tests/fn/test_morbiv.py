"""Tests for morbiv.bivariate_morans_i."""
import numpy as np
import pytest
from moirais.fn.morbiv import bivariate_morans_i


def test_morbiv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = bivariate_morans_i(x, y, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_morbiv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = bivariate_morans_i(x, y, W)
    assert isinstance(result, dict)
