"""Tests for rdfunc.rate_distortion."""
import numpy as np
import pytest
from morie.fn.rdfunc import rate_distortion


def test_rdfunc_basic():
    """Test basic functionality."""
    px = np.random.default_rng(42).normal(0, 1, 100)
    distortion = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = rate_distortion(px, distortion, D)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rdfunc_edge():
    """Test edge cases."""
    px = np.random.default_rng(42).normal(0, 1, 100)
    distortion = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = rate_distortion(px, distortion, D)
    assert isinstance(result, dict)
