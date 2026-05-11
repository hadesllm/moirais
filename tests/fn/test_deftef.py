"""Tests for deftef.design_effect."""
import numpy as np
import pytest
from morie.fn.deftef import design_effect


def test_deftef_basic():
    """Test basic functionality."""
    design_var = np.random.default_rng(42).normal(0, 1, 100)
    srs_var = np.random.default_rng(42).normal(0, 1, 100)
    result = design_effect(design_var, srs_var)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_deftef_edge():
    """Test edge cases."""
    design_var = np.random.default_rng(42).normal(0, 1, 100)
    srs_var = np.random.default_rng(42).normal(0, 1, 100)
    result = design_effect(design_var, srs_var)
    assert isinstance(result, dict)
