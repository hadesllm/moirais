"""Tests for gebvs.gebv_selection."""
import numpy as np
import pytest
from morie.fn.gebvs import gebv_selection


def test_gebvs_basic():
    """Test basic functionality."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    method = 'auto'
    result = gebv_selection(marker_matrix, y_train, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gebvs_edge():
    """Test edge cases."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    method = 'auto'
    result = gebv_selection(marker_matrix, y_train, method)
    assert isinstance(result, dict)
