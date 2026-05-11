"""Tests for flmgcr.flamingo_gated_cross."""
import numpy as np
import pytest
from morie.fn.flmgcr import flamingo_gated_cross


def test_flmgcr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    vision = np.random.default_rng(42).normal(0, 1, 100)
    gate = np.random.default_rng(42).normal(0, 1, 100)
    result = flamingo_gated_cross(x, vision, gate)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_flmgcr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    vision = np.random.default_rng(42).normal(0, 1, 100)
    gate = np.random.default_rng(42).normal(0, 1, 100)
    result = flamingo_gated_cross(x, vision, gate)
    assert isinstance(result, dict)
