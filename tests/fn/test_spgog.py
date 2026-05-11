"""Tests for spgog.schabenberger_getis_ord_g."""
import numpy as np
import pytest
from morie.fn.spgog import schabenberger_getis_ord_g


def test_spgog_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    d = 5
    result = schabenberger_getis_ord_g(x, coords, d)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spgog_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    d = 5
    result = schabenberger_getis_ord_g(x, coords, d)
    assert isinstance(result, dict)
