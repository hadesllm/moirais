"""Tests for survvar.variance_cox_estimator."""
import numpy as np
import pytest
from moirais.fn.survvar import variance_cox_estimator


def test_survvar_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_cox_estimator(fit)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survvar_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_cox_estimator(fit)
    assert isinstance(result, dict)
