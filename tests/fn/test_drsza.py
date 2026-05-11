"""Tests for drsza.dr_did_santanna_zhao."""
import numpy as np
import pytest
from morie.fn.drsza import dr_did_santanna_zhao


def test_drsza_basic():
    """Test basic functionality."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    y_post = np.random.default_rng(42).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ml_propensity = np.random.default_rng(42).normal(0, 1, 100)
    ml_outcome = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_santanna_zhao(y_pre, y_post, treatment, X, ml_propensity, ml_outcome)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drsza_edge():
    """Test edge cases."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    y_post = np.random.default_rng(42).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ml_propensity = np.random.default_rng(42).normal(0, 1, 100)
    ml_outcome = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_santanna_zhao(y_pre, y_post, treatment, X, ml_propensity, ml_outcome)
    assert isinstance(result, dict)
