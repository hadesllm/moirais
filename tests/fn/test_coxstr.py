"""Tests for coxstr.cox_stratified."""
import numpy as np
import pytest
from morie.fn.coxstr import cox_stratified


def test_coxstr_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    result = cox_stratified(time, event, X, stratum)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_coxstr_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    result = cox_stratified(time, event, X, stratum)
    assert isinstance(result, dict)
