"""Tests for hrzbwcv.horowitz_bw_cv_sim."""
import numpy as np
import pytest
from morie.fn.hrzbwcv import horowitz_bw_cv_sim


def test_hrzbwcv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_bw_cv_sim(x, y, beta_hat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzbwcv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_bw_cv_sim(x, y, beta_hat)
    assert isinstance(result, dict)
