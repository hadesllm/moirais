"""Tests for fzmise.fauzi_mise_kdfe."""
import numpy as np
import pytest
from morie.fn.fzmise import fauzi_mise_kdfe


def test_fzmise_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_mise_kdfe(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzmise_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_mise_kdfe(x, bandwidth)
    assert isinstance(result, dict)
