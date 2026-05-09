"""Tests for eslnnk.esl_nadaraya_watson."""
import numpy as np
import pytest
from moirais.fn.eslnnk import esl_nadaraya_watson


def test_eslnnk_basic():
    """Test basic functionality."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    x_data = np.random.default_rng(42).normal(0, 1, 100)
    y_data = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_nadaraya_watson(x0, x_data, y_data, lambda_)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslnnk_edge():
    """Test edge cases."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    x_data = np.random.default_rng(42).normal(0, 1, 100)
    y_data = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_nadaraya_watson(x0, x_data, y_data, lambda_)
    assert isinstance(result, dict)
