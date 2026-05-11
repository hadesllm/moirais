"""Tests for spnug.schabenberger_nugget_effect."""
import numpy as np
import pytest
from morie.fn.spnug import schabenberger_nugget_effect


def test_spnug_basic():
    """Test basic functionality."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_nugget_effect(h, nugget, sill, range)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spnug_edge():
    """Test edge cases."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_nugget_effect(h, nugget, sill, range)
    assert isinstance(result, dict)
