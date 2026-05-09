"""Tests for hmonnx.geron_onnx_export."""
import numpy as np
import pytest
from moirais.fn.hmonnx import geron_onnx_export


def test_hmonnx_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    args = np.random.default_rng(42).normal(0, 1, 100)
    file = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_onnx_export(model, args, file)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmonnx_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    args = np.random.default_rng(42).normal(0, 1, 100)
    file = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_onnx_export(model, args, file)
    assert isinstance(result, dict)
