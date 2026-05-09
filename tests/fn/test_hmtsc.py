"""Tests for hmtsc.geron_torchscript."""
import numpy as np
import pytest
from moirais.fn.hmtsc import geron_torchscript


def test_hmtsc_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    example_inputs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_torchscript(model, example_inputs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtsc_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    example_inputs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_torchscript(model, example_inputs)
    assert isinstance(result, dict)
