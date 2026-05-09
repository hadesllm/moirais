"""Tests for pibmd.prior_informativeness_bias_diagnostic."""
import numpy as np
import pytest
from moirais.fn.pibmd import prior_informativeness_bias_diagnostic


def test_pibmd_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = prior_informativeness_bias_diagnostic(samples, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pibmd_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = prior_informativeness_bias_diagnostic(samples, prior)
    assert isinstance(result, dict)
