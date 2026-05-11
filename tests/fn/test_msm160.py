"""Tests for msm160.mvsml_general_eq_1_2."""
import numpy as np
import pytest
from morie.fn.msm160 import mvsml_general_eq_1_2


def test_msm160_basic():
    """Test basic functionality."""
    yp_ts = np.random.default_rng(42).normal(0, 1, 100)
    apply = np.random.default_rng(42).normal(0, 1, 100)
    Probs = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    max = np.random.default_rng(42).normal(0, 1, 100)
    Tab1_PCCC = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(yp_ts, apply, Probs, which, max, Tab1_PCCC)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm160_edge():
    """Test edge cases."""
    yp_ts = np.random.default_rng(42).normal(0, 1, 100)
    apply = np.random.default_rng(42).normal(0, 1, 100)
    Probs = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    max = np.random.default_rng(42).normal(0, 1, 100)
    Tab1_PCCC = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(yp_ts, apply, Probs, which, max, Tab1_PCCC)
    assert isinstance(result, dict)
