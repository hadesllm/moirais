"""Tests for jotft.joseph_temporal_fusion_transformer."""
import numpy as np
import pytest
from morie.fn.jotft import joseph_temporal_fusion_transformer


def test_jotft_basic():
    """Test basic functionality."""
    static = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    known = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_temporal_fusion_transformer(static, observed, known, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jotft_edge():
    """Test edge cases."""
    static = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    known = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_temporal_fusion_transformer(static, observed, known, horizon)
    assert isinstance(result, dict)
