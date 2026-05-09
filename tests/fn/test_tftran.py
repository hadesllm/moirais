"""Tests for tftran.temporal_fusion_transformer."""
import numpy as np
import pytest
from moirais.fn.tftran import temporal_fusion_transformer


def test_tftran_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    static = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = temporal_fusion_transformer(y, X, static, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tftran_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    static = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = temporal_fusion_transformer(y, X, static, horizon)
    assert isinstance(result, dict)
