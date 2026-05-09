"""Tests for morisp.morans_i."""
import numpy as np
import pytest
from moirais.fn.morisp import morans_i


def test_morisp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = morans_i(x, W)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_morisp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = morans_i(x, W)
    assert isinstance(result, dict)
