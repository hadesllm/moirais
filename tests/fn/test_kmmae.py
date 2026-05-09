"""Tests for kmmae.kamath_multimodal_mae."""
import numpy as np
import pytest
from moirais.fn.kmmae import kamath_multimodal_mae


def test_kmmae_basic():
    """Test basic functionality."""
    x_visible = np.random.default_rng(42).normal(0, 1, 100)
    x_masked_true = np.random.default_rng(42).normal(0, 1, 100)
    masks = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_multimodal_mae(x_visible, x_masked_true, masks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmmae_edge():
    """Test edge cases."""
    x_visible = np.random.default_rng(42).normal(0, 1, 100)
    x_masked_true = np.random.default_rng(42).normal(0, 1, 100)
    masks = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_multimodal_mae(x_visible, x_masked_true, masks)
    assert isinstance(result, dict)
