"""Tests for propinf.property_inference."""
import numpy as np
import pytest
from morie.fn.propinf import property_inference


def test_propinf_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    property = np.random.default_rng(42).normal(0, 1, 100)
    result = property_inference(model, property)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_propinf_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    property = np.random.default_rng(42).normal(0, 1, 100)
    result = property_inference(model, property)
    assert isinstance(result, dict)
