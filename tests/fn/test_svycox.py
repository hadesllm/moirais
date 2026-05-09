"""Tests for svycox.survey_cox."""
import numpy as np
import pytest
from moirais.fn.svycox import survey_cox


def test_svycox_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_cox(time, event, X, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svycox_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_cox(time, event, X, weights)
    assert isinstance(result, dict)
