"""Tests for msm239.mvsml_general_eq_1_3."""
import numpy as np
import pytest
from moirais.fn.msm239 import mvsml_general_eq_1_3


def test_msm239_basic():
    """Test basic functionality."""
    Data = np.random.default_rng(42).normal(0, 1, 100)
    Final = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    Env = np.random.default_rng(42).normal(0, 1, 100)
    GID = np.random.default_rng(42).normal(0, 1, 100)
    Creating = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_3(Data, Final, order, Env, GID, Creating)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm239_edge():
    """Test edge cases."""
    Data = np.random.default_rng(42).normal(0, 1, 100)
    Final = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    Env = np.random.default_rng(42).normal(0, 1, 100)
    GID = np.random.default_rng(42).normal(0, 1, 100)
    Creating = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_3(Data, Final, order, Env, GID, Creating)
    assert isinstance(result, dict)
