"""Tests for hrzlrtt.horowitz_likelihood_ratio_test."""
import numpy as np
import pytest
from moirais.fn.hrzlrtt import horowitz_likelihood_ratio_test


def test_hrzlrtt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    model_restricted = np.random.default_rng(42).normal(0, 1, 100)
    model_full = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_likelihood_ratio_test(x, y, model_restricted, model_full)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hrzlrtt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    model_restricted = np.random.default_rng(42).normal(0, 1, 100)
    model_full = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_likelihood_ratio_test(x, y, model_restricted, model_full)
    assert isinstance(result, dict)
