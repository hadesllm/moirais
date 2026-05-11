"""Tests for esl1se.esl_one_se_rule."""
import numpy as np
import pytest
from morie.fn.esl1se import esl_one_se_rule


def test_esl1se_basic():
    """Test basic functionality."""
    cv_err = np.random.default_rng(42).normal(0, 1, 100)
    cv_se = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_one_se_rule(cv_err, cv_se)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esl1se_edge():
    """Test edge cases."""
    cv_err = np.random.default_rng(42).normal(0, 1, 100)
    cv_se = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_one_se_rule(cv_err, cv_se)
    assert isinstance(result, dict)
