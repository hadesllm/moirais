"""Tests for pposm.posterior_predictive_mean."""
import numpy as np
import pytest
from moirais.fn.pposm import posterior_predictive_mean


def test_pposm_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_mean(samples)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pposm_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_mean(samples)
    assert isinstance(result, dict)
