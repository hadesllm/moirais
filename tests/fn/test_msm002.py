"""Tests for msm002.mvsml_general_eq_1_2."""
import numpy as np
import pytest
from moirais.fn.msm002 import mvsml_general_eq_1_2


def test_msm002_basic():
    """Test basic functionality."""
    Data = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    one = np.random.default_rng(42).normal(0, 1, 100)
    way = np.random.default_rng(42).normal(0, 1, 100)
    classi = np.random.default_rng(42).normal(0, 1, 100)
    cation = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(Data, a, one, way, classi, cation)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm002_edge():
    """Test edge cases."""
    Data = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    one = np.random.default_rng(42).normal(0, 1, 100)
    way = np.random.default_rng(42).normal(0, 1, 100)
    classi = np.random.default_rng(42).normal(0, 1, 100)
    cation = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(Data, a, one, way, classi, cation)
    assert isinstance(result, dict)
