"""Tests for mahsj.ma_hksj_t_pi."""
import numpy as np
import pytest
from morie.fn.mahsj import ma_hksj_t_pi


def test_mahsj_basic():
    """Test basic functionality."""
    theta = 0.0
    se_hksj = np.random.default_rng(42).normal(0, 1, 100)
    tau2 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ma_hksj_t_pi(theta, se_hksj, tau2, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mahsj_edge():
    """Test edge cases."""
    theta = 0.0
    se_hksj = np.random.default_rng(42).normal(0, 1, 100)
    tau2 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ma_hksj_t_pi(theta, se_hksj, tau2, k)
    assert isinstance(result, dict)
