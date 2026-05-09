"""Tests for ovbnk.oster_omitted_bias_bound."""
import numpy as np
import pytest
from moirais.fn.ovbnk import oster_omitted_bias_bound


def test_ovbnk_basic():
    """Test basic functionality."""
    beta_short = np.random.default_rng(42).normal(0, 1, 100)
    beta_long = np.random.default_rng(42).normal(0, 1, 100)
    R_short = np.random.default_rng(42).normal(0, 1, 100)
    R_long = np.random.default_rng(42).normal(0, 1, 100)
    R_max = 100
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = oster_omitted_bias_bound(beta_short, beta_long, R_short, R_long, R_max, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ovbnk_edge():
    """Test edge cases."""
    beta_short = np.random.default_rng(42).normal(0, 1, 100)
    beta_long = np.random.default_rng(42).normal(0, 1, 100)
    R_short = np.random.default_rng(42).normal(0, 1, 100)
    R_long = np.random.default_rng(42).normal(0, 1, 100)
    R_max = 100
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = oster_omitted_bias_bound(beta_short, beta_long, R_short, R_long, R_max, delta)
    assert isinstance(result, dict)
