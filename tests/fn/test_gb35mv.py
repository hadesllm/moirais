"""Tests for gb35mv.gibbons_rvn_moments."""
import numpy as np
import pytest
from morie.fn.gb35mv import gibbons_rvn_moments


def test_gb35mv_basic():
    """Test basic functionality."""
    n = 100
    result = gibbons_rvn_moments(n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb35mv_edge():
    """Test edge cases."""
    n = 100
    result = gibbons_rvn_moments(n)
    assert isinstance(result, dict)
