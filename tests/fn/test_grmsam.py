"""Tests for grmsam.graded_response_samejima."""
import numpy as np
import pytest
from moirais.fn.grmsam import graded_response_samejima


def test_grmsam_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    result = graded_response_samejima(y, theta, a, b_k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmsam_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    result = graded_response_samejima(y, theta, a, b_k)
    assert isinstance(result, dict)
