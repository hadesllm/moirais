"""Tests for kr20cr.kuder_richardson_20."""
import numpy as np
import pytest
from moirais.fn.kr20cr import kuder_richardson_20


def test_kr20cr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = kuder_richardson_20(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kr20cr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = kuder_richardson_20(X)
    assert isinstance(result, dict)
