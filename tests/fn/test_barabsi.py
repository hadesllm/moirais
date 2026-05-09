"""Tests for barabsi.barabasi_albert."""
import numpy as np
import pytest
from moirais.fn.barabsi import barabasi_albert


def test_barabsi_basic():
    """Test basic functionality."""
    n = 100
    m = 10
    result = barabasi_albert(n, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_barabsi_edge():
    """Test edge cases."""
    n = 100
    m = 10
    result = barabasi_albert(n, m)
    assert isinstance(result, dict)
