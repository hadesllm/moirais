"""Tests for fpcadj.finite_population_corr."""
import numpy as np
import pytest
from morie.fn.fpcadj import finite_population_corr


def test_fpcadj_basic():
    """Test basic functionality."""
    n = 100
    N = 100
    result = finite_population_corr(n, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fpcadj_edge():
    """Test edge cases."""
    n = 100
    N = 100
    result = finite_population_corr(n, N)
    assert isinstance(result, dict)
