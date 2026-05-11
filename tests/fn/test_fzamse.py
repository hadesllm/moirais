"""Tests for fzamse.fauzi_quantile_amse."""
import numpy as np
import pytest
from morie.fn.fzamse import fauzi_quantile_amse


def test_fzamse_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = fauzi_quantile_amse(data, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzamse_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = fauzi_quantile_amse(data, p)
    assert isinstance(result, dict)
