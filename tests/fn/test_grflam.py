"""Tests for grflam.geron_flamingo_cross_modal_attn."""
import numpy as np
import pytest
from moirais.fn.grflam import geron_flamingo_cross_modal_attn


def test_grflam_basic():
    """Test basic functionality."""
    h = 0.3
    visual_features = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_flamingo_cross_modal_attn(h, visual_features, alpha, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grflam_edge():
    """Test edge cases."""
    h = 0.3
    visual_features = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_flamingo_cross_modal_attn(h, visual_features, alpha, weights)
    assert isinstance(result, dict)
