"""Tests for msm004.mvsml_general_eq_1_3."""
import numpy as np
import pytest
from morie.fn.msm004 import mvsml_general_eq_1_3


def test_msm004_basic():
    """Test basic functionality."""
    re = np.random.default_rng(42).normal(0, 1, 100)
    centered = np.random.default_rng(42).normal(0, 1, 100)
    around = np.random.default_rng(42).normal(0, 1, 100)
    zero = np.random.default_rng(42).normal(0, 1, 100)
    than = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_3(re, centered, around, zero, than, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm004_edge():
    """Test edge cases."""
    re = np.random.default_rng(42).normal(0, 1, 100)
    centered = np.random.default_rng(42).normal(0, 1, 100)
    around = np.random.default_rng(42).normal(0, 1, 100)
    zero = np.random.default_rng(42).normal(0, 1, 100)
    than = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_3(re, centered, around, zero, than, the)
    assert isinstance(result, dict)
