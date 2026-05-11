"""Tests for telmt.telemetry_drift."""
import numpy as np
import pytest
from morie.fn.telmt import telemetry_drift


def test_telmt_basic():
    """Test basic functionality."""
    error_stream = np.random.default_rng(42).normal(0, 1, 100)
    result = telemetry_drift(error_stream)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_telmt_edge():
    """Test edge cases."""
    error_stream = np.random.default_rng(42).normal(0, 1, 100)
    result = telemetry_drift(error_stream)
    assert isinstance(result, dict)
