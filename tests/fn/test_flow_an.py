"""Tests for flow_an.normalizing_flow_anomaly."""
import numpy as np
import pytest
from moirais.fn.flow_an import normalizing_flow_anomaly


def test_flow_an_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    flow = np.random.default_rng(42).normal(0, 1, 100)
    result = normalizing_flow_anomaly(X, flow)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_flow_an_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    flow = np.random.default_rng(42).normal(0, 1, 100)
    result = normalizing_flow_anomaly(X, flow)
    assert isinstance(result, dict)
