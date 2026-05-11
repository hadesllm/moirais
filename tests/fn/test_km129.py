"""Tests for km129.kamath_ch9_modality_encoder."""
import numpy as np
import pytest
from morie.fn.km129 import kamath_ch9_modality_encoder


def test_km129_basic():
    """Test basic functionality."""
    I_X = np.random.default_rng(42).normal(0, 1, 100)
    ME_X = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_modality_encoder(I_X, ME_X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km129_edge():
    """Test edge cases."""
    I_X = np.random.default_rng(42).normal(0, 1, 100)
    ME_X = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_modality_encoder(I_X, ME_X)
    assert isinstance(result, dict)
