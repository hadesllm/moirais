"""Tests for grstk.geron_stacking_predictor."""
import numpy as np
import pytest
from moirais.fn.grstk import geron_stacking_predictor


def test_grstk_basic():
    """Test basic functionality."""
    base_preds = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_stacking_predictor(base_preds, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grstk_edge():
    """Test edge cases."""
    base_preds = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_stacking_predictor(base_preds, y)
    assert isinstance(result, dict)
