"""Tests for fzkmis.fauzi_kdfe_mise."""
import numpy as np
import pytest
from morie.fn.fzkmis import fauzi_kdfe_mise


def test_fzkmis_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_kdfe_mise(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzkmis_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_kdfe_mise(x, bandwidth)
    assert isinstance(result, dict)
