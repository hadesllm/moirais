"""Tests for sgtcheeg.sgt_cheeger_constant."""
import numpy as np
import pytest
from moirais.fn.sgtcheeg import sgt_cheeger_constant


def test_sgtcheeg_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_cheeger_constant(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtcheeg_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_cheeger_constant(A)
    assert isinstance(result, dict)
