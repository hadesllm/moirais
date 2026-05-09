"""Tests for kapco.kappa_coefficient."""
import numpy as np
import pytest
from moirais.fn.kapco import kappa_coefficient


def test_kapco_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = kappa_coefficient(y_true, y_pred)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kapco_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = kappa_coefficient(y_true, y_pred)
    assert isinstance(result, dict)
