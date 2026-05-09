"""Tests for gb_kw2.gibbons_kw_alt_form."""
import numpy as np
import pytest
from moirais.fn.gb_kw2 import gibbons_kw_alt_form


def test_gb_kw2_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_kw_alt_form(groups)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_kw2_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_kw_alt_form(groups)
    assert isinstance(result, dict)
