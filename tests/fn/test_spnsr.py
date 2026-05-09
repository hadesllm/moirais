"""Tests for spnsr.schabenberger_nugget_sill_range_effect."""
import numpy as np
import pytest
from moirais.fn.spnsr import schabenberger_nugget_sill_range_effect


def test_spnsr_basic():
    """Test basic functionality."""
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    target_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_nugget_sill_range_effect(nugget, sill, range, target_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spnsr_edge():
    """Test edge cases."""
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    target_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_nugget_sill_range_effect(nugget, sill, range, target_dist)
    assert isinstance(result, dict)
