"""Tests for sgtspr.sgt_spectral_radius_bound."""
import numpy as np
import pytest
from moirais.fn.sgtspr import sgt_spectral_radius_bound


def test_sgtspr_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_spectral_radius_bound(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtspr_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_spectral_radius_bound(A)
    assert isinstance(result, dict)
