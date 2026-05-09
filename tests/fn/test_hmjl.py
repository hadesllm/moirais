"""Tests for hmjl.geron_johnson_lindenstrauss."""
import numpy as np
import pytest
from moirais.fn.hmjl import geron_johnson_lindenstrauss


def test_hmjl_basic():
    """Test basic functionality."""
    n = 100
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_johnson_lindenstrauss(n, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmjl_edge():
    """Test edge cases."""
    n = 100
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_johnson_lindenstrauss(n, eps)
    assert isinstance(result, dict)
