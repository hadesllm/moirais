"""Tests for grcos.geron_conv_output_size."""
import numpy as np
import pytest
from moirais.fn.grcos import geron_conv_output_size


def test_grcos_basic():
    """Test basic functionality."""
    in_size = 100
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    padding = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_conv_output_size(in_size, kernel, padding, stride)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grcos_edge():
    """Test edge cases."""
    in_size = 100
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    padding = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_conv_output_size(in_size, kernel, padding, stride)
    assert isinstance(result, dict)
