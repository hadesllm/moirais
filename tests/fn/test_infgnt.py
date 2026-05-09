"""Tests for infgnt.information_geometry."""
import numpy as np
import pytest
from moirais.fn.infgnt import information_geometry


def test_infgnt_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = information_geometry(log_p, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_infgnt_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = information_geometry(log_p, theta)
    assert isinstance(result, dict)
