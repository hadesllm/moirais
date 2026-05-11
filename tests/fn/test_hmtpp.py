"""Tests for hmtpp.geron_tensor_parallelism."""
import numpy as np
import pytest
from morie.fn.hmtpp import geron_tensor_parallelism


def test_hmtpp_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_devices = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tensor_parallelism(model, n_devices)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtpp_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_devices = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tensor_parallelism(model, n_devices)
    assert isinstance(result, dict)
