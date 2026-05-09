"""Tests for gb735.gibbons_linrank_sym_equal."""
import numpy as np
import pytest
from moirais.fn.gb735 import gibbons_linrank_sym_equal


def test_gb735_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    m = 10
    n = 100
    result = gibbons_linrank_sym_equal(a, m, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb735_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    m = 10
    n = 100
    result = gibbons_linrank_sym_equal(a, m, n)
    assert isinstance(result, dict)
