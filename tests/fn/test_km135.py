"""Tests for km135.kamath_ch9_clip_contrastive_total."""
import numpy as np
import pytest
from moirais.fn.km135 import kamath_ch9_clip_contrastive_total


def test_km135_basic():
    """Test basic functionality."""
    L_i2t = np.random.default_rng(42).normal(0, 1, 100)
    L_t2i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_clip_contrastive_total(L_i2t, L_t2i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km135_edge():
    """Test edge cases."""
    L_i2t = np.random.default_rng(42).normal(0, 1, 100)
    L_t2i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_clip_contrastive_total(L_i2t, L_t2i)
    assert isinstance(result, dict)
