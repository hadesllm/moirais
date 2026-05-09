"""Tests for genmol.generative_chemistry."""
import numpy as np
import pytest
from moirais.fn.genmol import generative_chemistry


def test_genmol_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_samples = np.random.default_rng(42).normal(0, 1, 100)
    conditions = np.random.default_rng(42).normal(0, 1, 100)
    result = generative_chemistry(model, n_samples, conditions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_genmol_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_samples = np.random.default_rng(42).normal(0, 1, 100)
    conditions = np.random.default_rng(42).normal(0, 1, 100)
    result = generative_chemistry(model, n_samples, conditions)
    assert isinstance(result, dict)
