"""Tests for memb.membership_inference."""
import numpy as np
import pytest
from morie.fn.memb import membership_inference


def test_memb_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    shadow_models = np.random.default_rng(42).normal(0, 1, 100)
    result = membership_inference(model, x, shadow_models)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_memb_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    shadow_models = np.random.default_rng(42).normal(0, 1, 100)
    result = membership_inference(model, x, shadow_models)
    assert isinstance(result, dict)
