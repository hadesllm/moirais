"""Tests for kmstst.kamath_stereoset_bias."""
import numpy as np
import pytest
from morie.fn.kmstst import kamath_stereoset_bias


def test_kmstst_basic():
    """Test basic functionality."""
    stereo_probs = np.random.default_rng(42).normal(0, 1, 100)
    anti_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_stereoset_bias(stereo_probs, anti_probs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmstst_edge():
    """Test edge cases."""
    stereo_probs = np.random.default_rng(42).normal(0, 1, 100)
    anti_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_stereoset_bias(stereo_probs, anti_probs)
    assert isinstance(result, dict)
