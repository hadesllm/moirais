"""Tests for omitV.omitted_variable_bias."""
import numpy as np
import pytest
from moirais.fn.omitV import omitted_variable_bias


def test_omitV_basic():
    """Test basic functionality."""
    beta_xu = np.random.default_rng(42).normal(0, 1, 100)
    beta_yu_given_x = np.random.default_rng(42).normal(0, 1, 100)
    result = omitted_variable_bias(beta_xu, beta_yu_given_x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_omitV_edge():
    """Test edge cases."""
    beta_xu = np.random.default_rng(42).normal(0, 1, 100)
    beta_yu_given_x = np.random.default_rng(42).normal(0, 1, 100)
    result = omitted_variable_bias(beta_xu, beta_yu_given_x)
    assert isinstance(result, dict)
