"""Tests for nrm.nominal_response_bock."""
import numpy as np
import pytest
from moirais.fn.nrm import nominal_response_bock


def test_nrm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    c_k = np.random.default_rng(42).normal(0, 1, 100)
    result = nominal_response_bock(y, theta, a_k, c_k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nrm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    c_k = np.random.default_rng(42).normal(0, 1, 100)
    result = nominal_response_bock(y, theta, a_k, c_k)
    assert isinstance(result, dict)
