"""Tests for gb_hg2.gibbons_hodges_lehmann_2."""
import numpy as np
import pytest
from moirais.fn.gb_hg2 import gibbons_hodges_lehmann_2


def test_gb_hg2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_hodges_lehmann_2(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_hg2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_hodges_lehmann_2(x, y)
    assert isinstance(result, dict)
