"""Tests for msm001.mvsml_general_eq_1_1."""
import numpy as np
import pytest
from moirais.fn.msm001 import mvsml_general_eq_1_1


def test_msm001_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    model = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    simpli = np.random.default_rng(42).normal(0, 1, 100)
    ed = np.random.default_rng(42).normal(0, 1, 100)
    description = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_1(A, model, a, simpli, ed, description)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm001_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    model = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    simpli = np.random.default_rng(42).normal(0, 1, 100)
    ed = np.random.default_rng(42).normal(0, 1, 100)
    description = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_1(A, model, a, simpli, ed, description)
    assert isinstance(result, dict)
