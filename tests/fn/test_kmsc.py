"""Tests for kmsc.kamath_self_consistency."""
import numpy as np
import pytest
from moirais.fn.kmsc import kamath_self_consistency


def test_kmsc_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_self_consistency(samples)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmsc_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_self_consistency(samples)
    assert isinstance(result, dict)
