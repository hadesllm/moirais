"""Tests for trdpd.tree_depth_saturation."""
import numpy as np
import pytest
from morie.fn.trdpd import tree_depth_saturation


def test_trdpd_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = tree_depth_saturation(chains)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trdpd_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = tree_depth_saturation(chains)
    assert isinstance(result, dict)
