"""Tests for watstro.watts_strogatz."""
import numpy as np
import pytest
from morie.fn.watstro import watts_strogatz


def test_watstro_basic():
    """Test basic functionality."""
    n = 100
    k = 5
    p = 5
    result = watts_strogatz(n, k, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_watstro_edge():
    """Test edge cases."""
    n = 100
    k = 5
    p = 5
    result = watts_strogatz(n, k, p)
    assert isinstance(result, dict)
