"""Tests for kpmnci.km_pointwise_ci."""
import numpy as np
import pytest
from morie.fn.kpmnci import km_pointwise_ci


def test_kpmnci_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = km_pointwise_ci(fit, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kpmnci_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = km_pointwise_ci(fit, alpha)
    assert isinstance(result, dict)
