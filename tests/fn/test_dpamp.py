"""Tests for dpamp.privacy_amplification."""
import numpy as np
import pytest
from morie.fn.dpamp import privacy_amplification


def test_dpamp_basic():
    """Test basic functionality."""
    epsilon = 1e-6
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = privacy_amplification(epsilon, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpamp_edge():
    """Test edge cases."""
    epsilon = 1e-6
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = privacy_amplification(epsilon, q)
    assert isinstance(result, dict)
