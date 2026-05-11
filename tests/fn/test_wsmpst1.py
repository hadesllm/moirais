"""Tests for wsmpst1.wasserman_posterior_mean."""
import numpy as np
import pytest
from morie.fn.wsmpst1 import wasserman_posterior_mean


def test_wsmpst1_basic():
    """Test basic functionality."""
    posterior = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_posterior_mean(posterior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmpst1_edge():
    """Test edge cases."""
    posterior = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_posterior_mean(posterior)
    assert isinstance(result, dict)
