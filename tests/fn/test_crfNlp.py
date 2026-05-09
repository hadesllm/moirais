"""Tests for crfNlp.crf_sequence."""
import numpy as np
import pytest
from moirais.fn.crfNlp import crf_sequence


def test_crfNlp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = crf_sequence(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crfNlp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = crf_sequence(X, y)
    assert isinstance(result, dict)
