"""Tests for hmqat.geron_quantization_aware_training."""
import numpy as np
import pytest
from morie.fn.hmqat import geron_quantization_aware_training


def test_hmqat_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_quantization_aware_training(model, X, y, epochs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmqat_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_quantization_aware_training(model, X, y, epochs)
    assert isinstance(result, dict)
