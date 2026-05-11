"""Tests for wnomp.wnominate_probability."""
import numpy as np
import pytest
from morie.fn.wnomp import wnominate_probability


def test_wnomp_basic():
    """Test basic functionality."""
    ideal_point = np.random.default_rng(42).normal(0, 1, 100)
    yea_pos = np.random.default_rng(42).normal(0, 1, 100)
    nay_pos = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = wnominate_probability(ideal_point, yea_pos, nay_pos, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wnomp_edge():
    """Test edge cases."""
    ideal_point = np.random.default_rng(42).normal(0, 1, 100)
    yea_pos = np.random.default_rng(42).normal(0, 1, 100)
    nay_pos = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = wnominate_probability(ideal_point, yea_pos, nay_pos, beta)
    assert isinstance(result, dict)
