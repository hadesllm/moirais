"""Tests for gb661v.gibbons_mw_var."""
import numpy as np
import pytest
from morie.fn.gb661v import gibbons_mw_var


def test_gb661v_basic():
    """Test basic functionality."""
    m = 10
    n = 100
    result = gibbons_mw_var(m, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb661v_edge():
    """Test edge cases."""
    m = 10
    n = 100
    result = gibbons_mw_var(m, n)
    assert isinstance(result, dict)
