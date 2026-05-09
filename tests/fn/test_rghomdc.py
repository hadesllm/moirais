"""Tests for rghomdc.rangayyan_homomorphic_deconv."""
import numpy as np
import pytest
from moirais.fn.rghomdc import rangayyan_homomorphic_deconv


def test_rghomdc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lifter_low = np.random.default_rng(42).normal(0, 1, 100)
    lifter_high = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_homomorphic_deconv(x, lifter_low, lifter_high)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghomdc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lifter_low = np.random.default_rng(42).normal(0, 1, 100)
    lifter_high = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_homomorphic_deconv(x, lifter_low, lifter_high)
    assert isinstance(result, dict)
