"""Tests for surdrl.survey_dr_estimator."""
import numpy as np
import pytest
from moirais.fn.surdrl import survey_dr_estimator


def test_surdrl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    sampling_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = survey_dr_estimator(y, D, X, sampling_weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_surdrl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    sampling_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = survey_dr_estimator(y, D, X, sampling_weights)
    assert isinstance(result, dict)
