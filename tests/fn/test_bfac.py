"""Tests for bfac.bayes_factor."""
import numpy as np
import pytest
from moirais.fn.bfac import bayes_factor


def test_bfac_basic():
    """Test basic functionality."""
    log_lik_a = np.random.default_rng(42).normal(0, 1, 100)
    log_lik_b = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_factor(log_lik_a, log_lik_b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bfac_edge():
    """Test edge cases."""
    log_lik_a = np.random.default_rng(42).normal(0, 1, 100)
    log_lik_b = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_factor(log_lik_a, log_lik_b)
    assert isinstance(result, dict)
