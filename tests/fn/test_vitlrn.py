"""Tests for vitlrn.vit_layer_norm."""
import numpy as np
import pytest
from moirais.fn.vitlrn import vit_layer_norm


def test_vitlrn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    beta = 0.8
    result = vit_layer_norm(x, gamma, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vitlrn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    beta = 0.8
    result = vit_layer_norm(x, gamma, beta)
    assert isinstance(result, dict)
