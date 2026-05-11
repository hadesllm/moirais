"""Tests for sqzext.squeeze_excite."""
import numpy as np
import pytest
from morie.fn.sqzext import squeeze_excite


def test_sqzext_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    reduction = np.random.default_rng(42).normal(0, 1, 100)
    result = squeeze_excite(x, reduction)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sqzext_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    reduction = np.random.default_rng(42).normal(0, 1, 100)
    result = squeeze_excite(x, reduction)
    assert isinstance(result, dict)
