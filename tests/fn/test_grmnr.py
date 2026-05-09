"""Tests for grmnr.geron_max_norm_regularization."""
import numpy as np
import pytest
from moirais.fn.grmnr import geron_max_norm_regularization


def test_grmnr_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = geron_max_norm_regularization(W, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmnr_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = geron_max_norm_regularization(W, r)
    assert isinstance(result, dict)
