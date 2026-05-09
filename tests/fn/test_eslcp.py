"""Tests for eslcp.esl_mallows_cp."""
import numpy as np
import pytest
from moirais.fn.eslcp import esl_mallows_cp


def test_eslcp_basic():
    """Test basic functionality."""
    RSS = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    n = 100
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_mallows_cp(RSS, d, n, sigma2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslcp_edge():
    """Test edge cases."""
    RSS = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    n = 100
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_mallows_cp(RSS, d, n, sigma2)
    assert isinstance(result, dict)
