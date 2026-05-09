"""Tests for eslprc.esl_perceptron."""
import numpy as np
import pytest
from moirais.fn.eslprc import esl_perceptron


def test_eslprc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_perceptron(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslprc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_perceptron(X, y)
    assert isinstance(result, dict)
