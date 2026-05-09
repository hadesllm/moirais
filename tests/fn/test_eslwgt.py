"""Tests for eslwgt.esl_weight_decay."""
import numpy as np
import pytest
from moirais.fn.eslwgt import esl_weight_decay


def test_eslwgt_basic():
    """Test basic functionality."""
    weights = np.random.default_rng(45).exponential(1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_weight_decay(weights, lambda_)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslwgt_edge():
    """Test edge cases."""
    weights = np.random.default_rng(45).exponential(1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_weight_decay(weights, lambda_)
    assert isinstance(result, dict)
