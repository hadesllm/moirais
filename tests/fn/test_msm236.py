"""Tests for msm236.mvsml_general_eq_1_3."""
import numpy as np
import pytest
from moirais.fn.msm236 import mvsml_general_eq_1_3


def test_msm236_basic():
    """Test basic functionality."""
    Creating = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    lines = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_3(Creating, the, design, matrix, of, lines)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm236_edge():
    """Test edge cases."""
    Creating = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    lines = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_3(Creating, the, design, matrix, of, lines)
    assert isinstance(result, dict)
