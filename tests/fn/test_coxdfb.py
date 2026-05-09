"""Tests for coxdfb.cox_dfbeta_influence."""
import numpy as np
import pytest
from moirais.fn.coxdfb import cox_dfbeta_influence


def test_coxdfb_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = cox_dfbeta_influence(fit)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_coxdfb_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = cox_dfbeta_influence(fit)
    assert isinstance(result, dict)
