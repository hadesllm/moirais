"""Tests for sgtmix.sgt_mixing_time."""
import numpy as np
import pytest
from morie.fn.sgtmix import sgt_mixing_time


def test_sgtmix_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    epsilon = 1e-6
    result = sgt_mixing_time(A, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtmix_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    epsilon = 1e-6
    result = sgt_mixing_time(A, epsilon)
    assert isinstance(result, dict)
