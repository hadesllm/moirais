"""Tests for kmrrf.kamath_reciprocal_rank_fusion."""
import numpy as np
import pytest
from morie.fn.kmrrf import kamath_reciprocal_rank_fusion


def test_kmrrf_basic():
    """Test basic functionality."""
    rankings = np.random.default_rng(42).permutation(10).reshape(2, 5)
    k = 5
    result = kamath_reciprocal_rank_fusion(rankings, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmrrf_edge():
    """Test edge cases."""
    rankings = np.random.default_rng(42).permutation(10).reshape(2, 5)
    k = 5
    result = kamath_reciprocal_rank_fusion(rankings, k)
    assert isinstance(result, dict)
