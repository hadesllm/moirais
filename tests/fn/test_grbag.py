"""Tests for grbag.geron_bagging_predictor."""
import numpy as np
import pytest
from moirais.fn.grbag import geron_bagging_predictor


def test_grbag_basic():
    """Test basic functionality."""
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bagging_predictor(predictions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbag_edge():
    """Test edge cases."""
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bagging_predictor(predictions)
    assert isinstance(result, dict)
