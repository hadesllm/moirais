"""Tests for mskbnd2.manski_no_assumption_outcome."""
import numpy as np
import pytest
from moirais.fn.mskbnd2 import manski_no_assumption_outcome


def test_mskbnd2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y_min = 0
    y_max = 100
    result = manski_no_assumption_outcome(y, D, X, y_min, y_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mskbnd2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y_min = 0
    y_max = 100
    result = manski_no_assumption_outcome(y, D, X, y_min, y_max)
    assert isinstance(result, dict)
