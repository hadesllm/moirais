"""Tests for msm005.mvsml_general_eq_1_4."""
import numpy as np
import pytest
from morie.fn.msm005 import mvsml_general_eq_1_4


def test_msm005_basic():
    """Test basic functionality."""
    mental = np.random.default_rng(42).normal(0, 1, 100)
    effects = np.random.default_rng(42).normal(0, 1, 100)
    Two = np.random.default_rng(42).normal(0, 1, 100)
    drawbacks = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_4(mental, effects, Two, drawbacks, of, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm005_edge():
    """Test edge cases."""
    mental = np.random.default_rng(42).normal(0, 1, 100)
    effects = np.random.default_rng(42).normal(0, 1, 100)
    Two = np.random.default_rng(42).normal(0, 1, 100)
    drawbacks = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_4(mental, effects, Two, drawbacks, of, the)
    assert isinstance(result, dict)
