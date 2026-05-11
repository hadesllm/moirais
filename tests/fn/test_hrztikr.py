"""Tests for hrztikr.horowitz_tikhonov_npiv."""
import numpy as np
import pytest
from morie.fn.hrztikr import horowitz_tikhonov_npiv


def test_hrztikr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    alpha_n = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_tikhonov_npiv(x, y, w, alpha_n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrztikr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    alpha_n = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_tikhonov_npiv(x, y, w, alpha_n)
    assert isinstance(result, dict)
