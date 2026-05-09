"""Tests for kappaw.weighted_kappa."""
import numpy as np
import pytest
from moirais.fn.kappaw import weighted_kappa


def test_kappaw_basic():
    """Test basic functionality."""
    rater1 = np.random.default_rng(42).normal(0, 1, 100)
    rater2 = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_kappa(rater1, rater2, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kappaw_edge():
    """Test edge cases."""
    rater1 = np.random.default_rng(42).normal(0, 1, 100)
    rater2 = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_kappa(rater1, rater2, weights)
    assert isinstance(result, dict)
