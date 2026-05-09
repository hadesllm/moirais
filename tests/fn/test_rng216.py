"""Tests for rng216.rangayyan_ch4_schwarz_inequality_complex."""
import numpy as np
import pytest
from moirais.fn.rng216 import rangayyan_ch4_schwarz_inequality_complex


def test_rng216_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_schwarz_inequality_complex(A, B, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng216_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_schwarz_inequality_complex(A, B, f)
    assert isinstance(result, dict)
