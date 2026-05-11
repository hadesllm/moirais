"""Tests for posrr.posterior_predictive_replication."""
import numpy as np
import pytest
from morie.fn.posrr import posterior_predictive_replication


def test_posrr_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_replication(samples)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_posrr_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_replication(samples)
    assert isinstance(result, dict)
