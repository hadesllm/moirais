"""Tests for mxetA.max_exceedance_curve."""
import numpy as np
import pytest
from moirais.fn.mxetA import max_exceedance_curve


def test_mxetA_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = max_exceedance_curve(coords, range)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mxetA_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = max_exceedance_curve(coords, range)
    assert isinstance(result, dict)
