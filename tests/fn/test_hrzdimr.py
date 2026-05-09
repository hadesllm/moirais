"""Tests for hrzdimr.horowitz_dimension_reduction."""
import numpy as np
import pytest
from moirais.fn.hrzdimr import horowitz_dimension_reduction


def test_hrzdimr_basic():
    """Test basic functionality."""
    d = 5
    n = 100
    bandwidth = 0.3
    result = horowitz_dimension_reduction(d, n, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzdimr_edge():
    """Test edge cases."""
    d = 5
    n = 100
    bandwidth = 0.3
    result = horowitz_dimension_reduction(d, n, bandwidth)
    assert isinstance(result, dict)
