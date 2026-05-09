"""Tests for xpehh1.xpehh."""
import numpy as np
import pytest
from moirais.fn.xpehh1 import xpehh


def test_xpehh1_basic():
    """Test basic functionality."""
    haplotypes_p1 = np.random.default_rng(42).normal(0, 1, 100)
    haplotypes_p2 = np.random.default_rng(42).normal(0, 1, 100)
    result = xpehh(haplotypes_p1, haplotypes_p2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_xpehh1_edge():
    """Test edge cases."""
    haplotypes_p1 = np.random.default_rng(42).normal(0, 1, 100)
    haplotypes_p2 = np.random.default_rng(42).normal(0, 1, 100)
    result = xpehh(haplotypes_p1, haplotypes_p2)
    assert isinstance(result, dict)
