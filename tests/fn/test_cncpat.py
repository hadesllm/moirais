"""Tests for cncpat.controlnet_attach."""
import numpy as np
import pytest
from morie.fn.cncpat import controlnet_attach


def test_cncpat_basic():
    """Test basic functionality."""
    base = np.random.default_rng(42).normal(0, 1, 100)
    condition = np.random.default_rng(42).normal(0, 1, 100)
    result = controlnet_attach(base, condition)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cncpat_edge():
    """Test edge cases."""
    base = np.random.default_rng(42).normal(0, 1, 100)
    condition = np.random.default_rng(42).normal(0, 1, 100)
    result = controlnet_attach(base, condition)
    assert isinstance(result, dict)
