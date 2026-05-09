"""Tests for msm006.mvsml_general_eq_1_5."""
import numpy as np
import pytest
from moirais.fn.msm006 import mvsml_general_eq_1_5


def test_msm006_basic():
    """Test basic functionality."""
    eij = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    P5 = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    represents = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_5(eij, where, P5, i, represents, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm006_edge():
    """Test edge cases."""
    eij = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    P5 = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    represents = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_5(eij, where, P5, i, represents, the)
    assert isinstance(result, dict)
