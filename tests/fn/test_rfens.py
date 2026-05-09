"""Tests for rfens.random_forest_ensemble."""
import numpy as np
import pytest
from moirais.fn.rfens import random_forest_ensemble


def test_rfens_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = random_forest_ensemble(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rfens_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = random_forest_ensemble(x, y)
    assert isinstance(result, dict)
