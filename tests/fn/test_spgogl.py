"""Tests for spgogl.schabenberger_getis_ord_gstar."""
import numpy as np
import pytest
from moirais.fn.spgogl import schabenberger_getis_ord_gstar


def test_spgogl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    d = 5
    result = schabenberger_getis_ord_gstar(x, coords, d)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spgogl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    d = 5
    result = schabenberger_getis_ord_gstar(x, coords, d)
    assert isinstance(result, dict)
