"""Tests for mafshz.ma_fishers_z."""
import numpy as np
import pytest
from morie.fn.mafshz import ma_fishers_z


def test_mafshz_basic():
    """Test basic functionality."""
    r = 10
    n = 100
    result = ma_fishers_z(r, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_mafshz_edge():
    """Test edge cases."""
    r = 10
    n = 100
    result = ma_fishers_z(r, n)
    assert isinstance(result, dict)
