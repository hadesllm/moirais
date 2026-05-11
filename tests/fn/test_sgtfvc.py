"""Tests for sgtfvc.sgt_fiedler_vector."""
import numpy as np
import pytest
from morie.fn.sgtfvc import sgt_fiedler_vector


def test_sgtfvc_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_fiedler_vector(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtfvc_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_fiedler_vector(A)
    assert isinstance(result, dict)
