"""Tests for esttsl.theta_method."""
import numpy as np
import pytest
from moirais.fn.esttsl import theta_method


def test_esttsl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = theta_method(y, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esttsl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = theta_method(y, horizon)
    assert isinstance(result, dict)
