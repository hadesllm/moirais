"""Tests for vrmed.variance_based_mediation."""
import numpy as np
import pytest
from moirais.fn.vrmed import variance_based_mediation


def test_vrmed_basic():
    """Test basic functionality."""
    r2_full = np.random.default_rng(42).normal(0, 1, 100)
    r2_partial = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_based_mediation(r2_full, r2_partial)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vrmed_edge():
    """Test edge cases."""
    r2_full = np.random.default_rng(42).normal(0, 1, 100)
    r2_partial = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_based_mediation(r2_full, r2_partial)
    assert isinstance(result, dict)
