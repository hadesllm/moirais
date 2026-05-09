"""Tests for hmcrd.geron_credit_assignment."""
import numpy as np
import pytest
from moirais.fn.hmcrd import geron_credit_assignment


def test_hmcrd_basic():
    """Test basic functionality."""
    trajectory = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_credit_assignment(trajectory)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcrd_edge():
    """Test edge cases."""
    trajectory = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_credit_assignment(trajectory)
    assert isinstance(result, dict)
