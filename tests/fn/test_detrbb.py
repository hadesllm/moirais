"""Tests for detrbb.detr_set_prediction."""
import numpy as np
import pytest
from morie.fn.detrbb import detr_set_prediction


def test_detrbb_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    n_objects = np.random.default_rng(42).normal(0, 1, 100)
    result = detr_set_prediction(image, queries, n_objects)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_detrbb_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    n_objects = np.random.default_rng(42).normal(0, 1, 100)
    result = detr_set_prediction(image, queries, n_objects)
    assert isinstance(result, dict)
