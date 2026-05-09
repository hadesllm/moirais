"""Tests for hmmpp.geron_model_parallelism."""
import numpy as np
import pytest
from moirais.fn.hmmpp import geron_model_parallelism


def test_hmmpp_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_devices = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_model_parallelism(model, n_devices)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmpp_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_devices = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_model_parallelism(model, n_devices)
    assert isinstance(result, dict)
