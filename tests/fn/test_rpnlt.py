"""Tests for rpnlt.roughness_penalty."""
import numpy as np
import pytest
from morie.fn.rpnlt import roughness_penalty


def test_rpnlt_basic():
    """Test basic functionality."""
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = roughness_penalty(basis, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rpnlt_edge():
    """Test edge cases."""
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = roughness_penalty(basis, lam)
    assert isinstance(result, dict)
