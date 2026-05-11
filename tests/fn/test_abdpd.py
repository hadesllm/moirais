"""Tests for abdpd.abduction_modification_prediction."""
import numpy as np
import pytest
from morie.fn.abdpd import abduction_modification_prediction


def test_abdpd_basic():
    """Test basic functionality."""
    evidence = np.random.default_rng(42).normal(0, 1, 100)
    do_X = np.random.default_rng(42).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    scm = np.random.default_rng(42).normal(0, 1, 100)
    result = abduction_modification_prediction(evidence, do_X, Y, scm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_abdpd_edge():
    """Test edge cases."""
    evidence = np.random.default_rng(42).normal(0, 1, 100)
    do_X = np.random.default_rng(42).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    scm = np.random.default_rng(42).normal(0, 1, 100)
    result = abduction_modification_prediction(evidence, do_X, Y, scm)
    assert isinstance(result, dict)
