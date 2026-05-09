"""Tests for ghcls.ghosal_np_classification."""
import numpy as np
import pytest
from moirais.fn.ghcls import ghosal_np_classification


def test_ghcls_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_np_classification(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghcls_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_np_classification(x, y)
    assert isinstance(result, dict)
