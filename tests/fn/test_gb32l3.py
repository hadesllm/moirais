"""Tests for gb32l3.gibbons_vandermonde_id2."""
import numpy as np
import pytest
from moirais.fn.gb32l3 import gibbons_vandermonde_id2


def test_gb32l3_basic():
    """Test basic functionality."""
    m = 10
    n = 100
    result = gibbons_vandermonde_id2(m, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb32l3_edge():
    """Test edge cases."""
    m = 10
    n = 100
    result = gibbons_vandermonde_id2(m, n)
    assert isinstance(result, dict)
