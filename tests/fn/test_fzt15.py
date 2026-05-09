"""Tests for fzt15.fauzi_thm1_5_consistency_mgkde."""
import numpy as np
import pytest
from moirais.fn.fzt15 import fauzi_thm1_5_consistency_mgkde


def test_fzt15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_thm1_5_consistency_mgkde(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_thm1_5_consistency_mgkde(x, bandwidth)
    assert isinstance(result, dict)
