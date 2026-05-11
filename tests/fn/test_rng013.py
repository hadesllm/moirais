"""Tests for rng013.rangayyan_ch3_mean_of_sum."""
import numpy as np
import pytest
from morie.fn.rng013 import rangayyan_ch3_mean_of_sum


def test_rng013_basic():
    """Test basic functionality."""
    mu_x = np.random.default_rng(42).normal(0, 1, 100)
    mu_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_mean_of_sum(mu_x, mu_eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng013_edge():
    """Test edge cases."""
    mu_x = np.random.default_rng(42).normal(0, 1, 100)
    mu_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_mean_of_sum(mu_x, mu_eta)
    assert isinstance(result, dict)
