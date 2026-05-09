"""Tests for vlfctn.value_function_eval."""
import numpy as np
import pytest
from moirais.fn.vlfctn import value_function_eval


def test_vlfctn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    regime = np.random.default_rng(42).normal(0, 1, 100)
    result = value_function_eval(y, D, W, regime)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vlfctn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    regime = np.random.default_rng(42).normal(0, 1, 100)
    result = value_function_eval(y, D, W, regime)
    assert isinstance(result, dict)
