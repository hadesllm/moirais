"""Tests for chrnff.chernoff_bound."""
import numpy as np
import pytest
from moirais.fn.chrnff import chernoff_bound


def test_chrnff_basic():
    """Test basic functionality."""
    mgf = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = chernoff_bound(mgf, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chrnff_edge():
    """Test edge cases."""
    mgf = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = chernoff_bound(mgf, threshold)
    assert isinstance(result, dict)
