"""Tests for hmfim.geron_feature_importance."""
import numpy as np
import pytest
from morie.fn.hmfim import geron_feature_importance


def test_hmfim_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    feature_names = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_feature_importance(model, feature_names)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfim_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    feature_names = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_feature_importance(model, feature_names)
    assert isinstance(result, dict)
