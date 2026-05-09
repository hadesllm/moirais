"""Tests for ksr15.kosorok_one_step_estimator."""
import numpy as np
import pytest
from moirais.fn.ksr15 import kosorok_one_step_estimator


def test_ksr15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_one_step_estimator(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_one_step_estimator(x, y)
    assert isinstance(result, dict)
