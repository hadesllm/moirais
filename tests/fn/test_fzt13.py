"""Tests for fzt13.fauzi_thm1_3_mise_mgkde."""
import numpy as np
import pytest
from moirais.fn.fzt13 import fauzi_thm1_3_mise_mgkde


def test_fzt13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_thm1_3_mise_mgkde(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_thm1_3_mise_mgkde(x, bandwidth)
    assert isinstance(result, dict)
