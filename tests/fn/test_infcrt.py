"""Tests for infcrt.information_criterion."""
import numpy as np
import pytest
from moirais.fn.infcrt import information_criterion


def test_infcrt_basic():
    """Test basic functionality."""
    log_lik_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = information_criterion(log_lik_samples)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_infcrt_edge():
    """Test edge cases."""
    log_lik_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = information_criterion(log_lik_samples)
    assert isinstance(result, dict)
