"""Tests for sgtest.sgt_estrada_index."""
import numpy as np
import pytest
from moirais.fn.sgtest import sgt_estrada_index


def test_sgtest_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_estrada_index(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtest_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_estrada_index(A)
    assert isinstance(result, dict)
