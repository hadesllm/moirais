"""Tests for pcmpr1.prediction_compression."""
import numpy as np
import pytest
from moirais.fn.pcmpr1 import prediction_compression


def test_pcmpr1_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = prediction_compression(model, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pcmpr1_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = prediction_compression(model, data)
    assert isinstance(result, dict)
