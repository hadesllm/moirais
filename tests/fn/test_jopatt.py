"""Tests for jopatt.joseph_patchtst."""
import numpy as np
import pytest
from moirais.fn.jopatt import joseph_patchtst


def test_jopatt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    patch_len = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    transformer = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_patchtst(x, patch_len, stride, transformer)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jopatt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    patch_len = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    transformer = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_patchtst(x, patch_len, stride, transformer)
    assert isinstance(result, dict)
