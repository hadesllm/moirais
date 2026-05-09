"""Tests for hmsvm2.geron_save_load_pytorch."""
import numpy as np
import pytest
from moirais.fn.hmsvm2 import geron_save_load_pytorch


def test_hmsvm2_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_save_load_pytorch(model, path)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsvm2_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_save_load_pytorch(model, path)
    assert isinstance(result, dict)
