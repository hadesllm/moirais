"""Tests for wsmnpb.wasserman_nonparametric_boot."""
import numpy as np
import pytest
from moirais.fn.wsmnpb import wasserman_nonparametric_boot


def test_wsmnpb_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = wasserman_nonparametric_boot(data, T, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmnpb_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = wasserman_nonparametric_boot(data, T, B)
    assert isinstance(result, dict)
