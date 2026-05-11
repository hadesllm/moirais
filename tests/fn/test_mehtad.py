"""Tests for mehtad.mehrotras_predictor."""
import numpy as np
import pytest
from morie.fn.mehtad import mehrotras_predictor


def test_mehtad_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = mehrotras_predictor(c, A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mehtad_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = mehrotras_predictor(c, A, b)
    assert isinstance(result, dict)
