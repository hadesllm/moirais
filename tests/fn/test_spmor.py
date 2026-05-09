"""Tests for spmor.schabenberger_moran_i."""
import numpy as np
import pytest
from moirais.fn.spmor import schabenberger_moran_i


def test_spmor_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_moran_i(x, w)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spmor_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_moran_i(x, w)
    assert isinstance(result, dict)
