"""Tests for cfafit.cfa_fit_indices."""
import numpy as np
import pytest
from moirais.fn.cfafit import cfa_fit_indices


def test_cfafit_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = cfa_fit_indices(fit)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cfafit_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = cfa_fit_indices(fit)
    assert isinstance(result, dict)
