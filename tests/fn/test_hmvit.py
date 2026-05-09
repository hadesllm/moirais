"""Tests for hmvit.geron_vision_transformer."""
import numpy as np
import pytest
from moirais.fn.hmvit import geron_vision_transformer


def test_hmvit_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vision_transformer(image, patch_size, n_layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvit_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vision_transformer(image, patch_size, n_layers)
    assert isinstance(result, dict)
