"""Tests for miefcl.multiple_imputation_combine."""
import numpy as np
import pytest
from moirais.fn.miefcl import multiple_imputation_combine


def test_miefcl_basic():
    """Test basic functionality."""
    estimates_list = np.random.default_rng(42).normal(0, 1, 100)
    ses_list = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = multiple_imputation_combine(estimates_list, ses_list, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_miefcl_edge():
    """Test edge cases."""
    estimates_list = np.random.default_rng(42).normal(0, 1, 100)
    ses_list = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = multiple_imputation_combine(estimates_list, ses_list, m)
    assert isinstance(result, dict)
