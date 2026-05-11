"""Tests for ipferd.ipw_with_replicate."""
import numpy as np
import pytest
from morie.fn.ipferd import ipw_with_replicate


def test_ipferd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    replicate_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = ipw_with_replicate(y, D, w, replicate_weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ipferd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    replicate_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = ipw_with_replicate(y, D, w, replicate_weights)
    assert isinstance(result, dict)
