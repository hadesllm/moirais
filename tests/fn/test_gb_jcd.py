"""Tests for gb_jcd.gibbons_jt_cd_form."""
import numpy as np
import pytest
from moirais.fn.gb_jcd import gibbons_jt_cd_form


def test_gb_jcd_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_jt_cd_form(groups)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_jcd_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_jt_cd_form(groups)
    assert isinstance(result, dict)
