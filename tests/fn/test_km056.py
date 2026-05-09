"""Tests for km056.kamath_ch4_full_finetune_obj."""
import numpy as np
import pytest
from moirais.fn.km056 import kamath_ch4_full_finetune_obj


def test_km056_basic():
    """Test basic functionality."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch4_full_finetune_obj(Phi, x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km056_edge():
    """Test edge cases."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch4_full_finetune_obj(Phi, x, y)
    assert isinstance(result, dict)
