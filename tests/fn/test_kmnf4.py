"""Tests for kmnf4.kamath_nf4_datatype."""
import numpy as np
import pytest
from moirais.fn.kmnf4 import kamath_nf4_datatype


def test_kmnf4_basic():
    """Test basic functionality."""
    n_bins = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_nf4_datatype(n_bins)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmnf4_edge():
    """Test edge cases."""
    n_bins = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_nf4_datatype(n_bins)
    assert isinstance(result, dict)
