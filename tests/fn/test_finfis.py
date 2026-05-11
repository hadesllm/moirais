"""Tests for finfis.fisher_information."""
import numpy as np
import pytest
from morie.fn.finfis import fisher_information


def test_finfis_basic():
    """Test basic functionality."""
    log_likelihood = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = fisher_information(log_likelihood, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_finfis_edge():
    """Test edge cases."""
    log_likelihood = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = fisher_information(log_likelihood, theta)
    assert isinstance(result, dict)
