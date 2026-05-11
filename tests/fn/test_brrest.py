"""Tests for brrest.brr_balanced."""
import numpy as np
import pytest
from morie.fn.brrest import brr_balanced


def test_brrest_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    strata = np.random.default_rng(42).normal(0, 1, 100)
    brr_design = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_balanced(data, strata, brr_design)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_brrest_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    strata = np.random.default_rng(42).normal(0, 1, 100)
    brr_design = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_balanced(data, strata, brr_design)
    assert isinstance(result, dict)
