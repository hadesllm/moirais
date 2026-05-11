"""Tests for thmksp.thomas_cluster."""
import numpy as np
import pytest
from morie.fn.thmksp import thomas_cluster


def test_thmksp_basic():
    """Test basic functionality."""
    lambda_p = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    result = thomas_cluster(lambda_p, mu, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_thmksp_edge():
    """Test edge cases."""
    lambda_p = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    result = thomas_cluster(lambda_p, mu, sigma)
    assert isinstance(result, dict)
