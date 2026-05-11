"""Tests for klmsm1.kl_molecular_smooth."""
import numpy as np
import pytest
from morie.fn.klmsm1 import kl_molecular_smooth


def test_klmsm1_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = kl_molecular_smooth(p, q, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_klmsm1_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = kl_molecular_smooth(p, q, eps)
    assert isinstance(result, dict)
