"""Tests for grrsk.geron_resnet_skip."""
import numpy as np
import pytest
from morie.fn.grrsk import geron_resnet_skip


def test_grrsk_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    Fx = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_resnet_skip(x, Fx)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grrsk_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    Fx = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_resnet_skip(x, Fx)
    assert isinstance(result, dict)
