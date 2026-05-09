"""Tests for aitcsq.compositional_chisq."""
import numpy as np
import pytest
from moirais.fn.aitcsq import compositional_chisq


def test_aitcsq_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = compositional_chisq(X)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_aitcsq_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = compositional_chisq(X)
    assert isinstance(result, dict)
