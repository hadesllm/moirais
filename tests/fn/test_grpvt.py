"""Tests for grpvt.geron_pyramid_vit_stage."""
import numpy as np
import pytest
from moirais.fn.grpvt import geron_pyramid_vit_stage


def test_grpvt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    reduction_ratio = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pyramid_vit_stage(X, WQ, WK, WV, reduction_ratio)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grpvt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    reduction_ratio = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pyramid_vit_stage(X, WQ, WK, WV, reduction_ratio)
    assert isinstance(result, dict)
