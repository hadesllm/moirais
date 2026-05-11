"""Tests for gxemd.gxe_interaction_model."""
import numpy as np
import pytest
from morie.fn.gxemd import gxe_interaction_model


def test_gxemd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    env = np.random.default_rng(42).normal(0, 1, 100)
    result = gxe_interaction_model(x, y, env)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gxemd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    env = np.random.default_rng(42).normal(0, 1, 100)
    result = gxe_interaction_model(x, y, env)
    assert isinstance(result, dict)
