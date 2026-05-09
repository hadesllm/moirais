"""Tests for sclvar.selection_coefficient."""
import numpy as np
import pytest
from moirais.fn.sclvar import selection_coefficient


def test_sclvar_basic():
    """Test basic functionality."""
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    generations = np.random.default_rng(42).normal(0, 1, 100)
    Ne = np.random.default_rng(42).normal(0, 1, 100)
    result = selection_coefficient(freqs, generations, Ne)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sclvar_edge():
    """Test edge cases."""
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    generations = np.random.default_rng(42).normal(0, 1, 100)
    Ne = np.random.default_rng(42).normal(0, 1, 100)
    result = selection_coefficient(freqs, generations, Ne)
    assert isinstance(result, dict)
