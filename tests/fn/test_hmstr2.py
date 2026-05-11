"""Tests for hmstr2.geron_stride."""
import numpy as np
import pytest
from morie.fn.hmstr2 import geron_stride


def test_hmstr2_basic():
    """Test basic functionality."""
    in_dim = 2
    k = 5
    p = 5
    s = 90
    result = geron_stride(in_dim, k, p, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmstr2_edge():
    """Test edge cases."""
    in_dim = 2
    k = 5
    p = 5
    s = 90
    result = geron_stride(in_dim, k, p, s)
    assert isinstance(result, dict)
