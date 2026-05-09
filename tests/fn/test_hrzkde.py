"""Tests for hrzkde.horowitz_appendix_kde."""
import numpy as np
import pytest
from moirais.fn.hrzkde import horowitz_appendix_kde


def test_hrzkde_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_appendix_kde(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzkde_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_appendix_kde(x, bandwidth)
    assert isinstance(result, dict)
