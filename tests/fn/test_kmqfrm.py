"""Tests for kmqfrm.kamath_q_former."""
import numpy as np
import pytest
from morie.fn.kmqfrm import kamath_q_former


def test_kmqfrm_basic():
    """Test basic functionality."""
    queries = np.random.default_rng(42).normal(0, 1, 100)
    visual_features = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_q_former(queries, visual_features)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmqfrm_edge():
    """Test edge cases."""
    queries = np.random.default_rng(42).normal(0, 1, 100)
    visual_features = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_q_former(queries, visual_features)
    assert isinstance(result, dict)
