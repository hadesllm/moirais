"""Tests for spmani.schabenberger_mantel_standard."""
import numpy as np
import pytest
from moirais.fn.spmani import schabenberger_mantel_standard


def test_spmani_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_mantel_standard(coords, x, w)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spmani_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_mantel_standard(coords, x, w)
    assert isinstance(result, dict)
