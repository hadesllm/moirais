"""Tests for johanc.johansen_cointegration."""
import numpy as np
import pytest
from morie.fn.johanc import johansen_cointegration


def test_johanc_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = johansen_cointegration(Y, p)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_johanc_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = johansen_cointegration(Y, p)
    assert isinstance(result, dict)
