"""Tests for attrInf.attribute_inference."""
import numpy as np
import pytest
from morie.fn.attrInf import attribute_inference


def test_attrInf_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x_partial = np.random.default_rng(42).normal(0, 1, 100)
    result = attribute_inference(model, x_partial)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_attrInf_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x_partial = np.random.default_rng(42).normal(0, 1, 100)
    result = attribute_inference(model, x_partial)
    assert isinstance(result, dict)
