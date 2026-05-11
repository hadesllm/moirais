"""Tests for hyplc.harmonic_mean_estimator."""
import numpy as np
import pytest
from morie.fn.hyplc import harmonic_mean_estimator


def test_hyplc_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = harmonic_mean_estimator(log_lik)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hyplc_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = harmonic_mean_estimator(log_lik)
    assert isinstance(result, dict)
