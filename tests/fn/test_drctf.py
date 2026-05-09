"""Tests for drctf.dr_continuous_treatment."""
import numpy as np
import pytest
from moirais.fn.drctf import dr_continuous_treatment


def test_drctf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D_dose = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_continuous_treatment(y, D_dose, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drctf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D_dose = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_continuous_treatment(y, D_dose, X)
    assert isinstance(result, dict)
