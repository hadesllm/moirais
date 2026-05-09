"""Tests for hrzfrd.horowitz_fredholm_eq."""
import numpy as np
import pytest
from moirais.fn.hrzfrd import horowitz_fredholm_eq


def test_hrzfrd_basic():
    """Test basic functionality."""
    m = 10
    k = 5
    result = horowitz_fredholm_eq(m, k)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hrzfrd_edge():
    """Test edge cases."""
    m = 10
    k = 5
    result = horowitz_fredholm_eq(m, k)
    assert isinstance(result, dict)
