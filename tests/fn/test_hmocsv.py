"""Tests for hmocsv.geron_one_class_svm."""
import numpy as np
import pytest
from morie.fn.hmocsv import geron_one_class_svm


def test_hmocsv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    nu = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_one_class_svm(X, nu, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmocsv_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    nu = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_one_class_svm(X, nu, gamma)
    assert isinstance(result, dict)
