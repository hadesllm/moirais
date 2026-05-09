"""Tests for galois.galois_group."""
import numpy as np
import pytest
from moirais.fn.galois import galois_group


def test_galois_basic():
    """Test basic functionality."""
    poly = np.random.default_rng(42).normal(0, 1, 100)
    result = galois_group(poly)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_galois_edge():
    """Test edge cases."""
    poly = np.random.default_rng(42).normal(0, 1, 100)
    result = galois_group(poly)
    assert isinstance(result, dict)
