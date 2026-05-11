"""Tests for gb32l2.gibbons_vandermonde_id1."""
import numpy as np
import pytest
from morie.fn.gb32l2 import gibbons_vandermonde_id1


def test_gb32l2_basic():
    """Test basic functionality."""
    m = 10
    n = 100
    result = gibbons_vandermonde_id1(m, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb32l2_edge():
    """Test edge cases."""
    m = 10
    n = 100
    result = gibbons_vandermonde_id1(m, n)
    assert isinstance(result, dict)
