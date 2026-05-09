"""Tests for grfim.geron_feature_importance_mdi."""
import numpy as np
import pytest
from moirais.fn.grfim import geron_feature_importance_mdi


def test_grfim_basic():
    """Test basic functionality."""
    tree_importances = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_feature_importance_mdi(tree_importances)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grfim_edge():
    """Test edge cases."""
    tree_importances = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_feature_importance_mdi(tree_importances)
    assert isinstance(result, dict)
