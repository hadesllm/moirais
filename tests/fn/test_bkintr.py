"""Tests for bkintr.burkov_ngram_interpolation."""
import numpy as np
import pytest
from moirais.fn.bkintr import burkov_ngram_interpolation


def test_bkintr_basic():
    """Test basic functionality."""
    probs_by_order = np.random.default_rng(42).normal(0, 1, 100)
    lambdas = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_ngram_interpolation(probs_by_order, lambdas)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkintr_edge():
    """Test edge cases."""
    probs_by_order = np.random.default_rng(42).normal(0, 1, 100)
    lambdas = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_ngram_interpolation(probs_by_order, lambdas)
    assert isinstance(result, dict)
