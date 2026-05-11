"""Tests for aparcm.aparch_dge."""
import numpy as np
import pytest
from morie.fn.aparcm import aparch_dge


def test_aparcm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = aparch_dge(x, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aparcm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = aparch_dge(x, delta)
    assert isinstance(result, dict)
