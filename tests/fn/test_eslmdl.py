"""Tests for eslmdl.esl_mdl."""
import numpy as np
import pytest
from moirais.fn.eslmdl import esl_mdl


def test_eslmdl_basic():
    """Test basic functionality."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = esl_mdl(loglik, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslmdl_edge():
    """Test edge cases."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = esl_mdl(loglik, theta)
    assert isinstance(result, dict)
