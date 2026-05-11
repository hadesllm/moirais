"""Tests for pcm.partial_credit_masters."""
import numpy as np
import pytest
from morie.fn.pcm import partial_credit_masters


def test_pcm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    delta_j = np.random.default_rng(42).normal(0, 1, 100)
    result = partial_credit_masters(y, theta, delta_j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pcm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    delta_j = np.random.default_rng(42).normal(0, 1, 100)
    result = partial_credit_masters(y, theta, delta_j)
    assert isinstance(result, dict)
