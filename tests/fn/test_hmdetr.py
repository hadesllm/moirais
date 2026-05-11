"""Tests for hmdetr.geron_detr."""
import numpy as np
import pytest
from morie.fn.hmdetr import geron_detr


def test_hmdetr_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    n_queries = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_detr(image, n_queries)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdetr_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    n_queries = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_detr(image, n_queries)
    assert isinstance(result, dict)
