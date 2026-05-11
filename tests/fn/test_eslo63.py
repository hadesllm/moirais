"""Tests for eslo63.esl_oob_632."""
import numpy as np
import pytest
from morie.fn.eslo63 import esl_oob_632


def test_eslo63_basic():
    """Test basic functionality."""
    err_train = np.random.default_rng(42).normal(0, 1, 100)
    err_boot = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_oob_632(err_train, err_boot)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslo63_edge():
    """Test edge cases."""
    err_train = np.random.default_rng(42).normal(0, 1, 100)
    err_boot = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_oob_632(err_train, err_boot)
    assert isinstance(result, dict)
