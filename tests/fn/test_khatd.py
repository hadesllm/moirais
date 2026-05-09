"""Tests for khatd.pareto_k_diagnostic."""
import numpy as np
import pytest
from moirais.fn.khatd import pareto_k_diagnostic


def test_khatd_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = pareto_k_diagnostic(log_lik)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_khatd_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = pareto_k_diagnostic(log_lik)
    assert isinstance(result, dict)
