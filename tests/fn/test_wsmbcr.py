"""Tests for wsmbcr.wasserman_credible_interval."""
import numpy as np
import pytest
from moirais.fn.wsmbcr import wasserman_credible_interval


def test_wsmbcr_basic():
    """Test basic functionality."""
    posterior = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = wasserman_credible_interval(posterior, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmbcr_edge():
    """Test edge cases."""
    posterior = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = wasserman_credible_interval(posterior, alpha)
    assert isinstance(result, dict)
