"""Tests for hmmdc.geron_mode_collapse."""
import numpy as np
import pytest
from morie.fn.hmmdc import geron_mode_collapse


def test_hmmdc_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mode_collapse(samples)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmdc_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mode_collapse(samples)
    assert isinstance(result, dict)
