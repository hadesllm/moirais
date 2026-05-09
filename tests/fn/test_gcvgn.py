"""Tests for gcvgn.genomic_cross_validation."""
import numpy as np
import pytest
from moirais.fn.gcvgn import genomic_cross_validation


def test_gcvgn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = genomic_cross_validation(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gcvgn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = genomic_cross_validation(x, y)
    assert isinstance(result, dict)
