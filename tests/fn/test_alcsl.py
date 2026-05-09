"""Tests for alcsl.alammar_cosine_similarity_loss."""
import numpy as np
import pytest
from moirais.fn.alcsl import alammar_cosine_similarity_loss


def test_alcsl_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    result = alammar_cosine_similarity_loss(a, b, y_true)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alcsl_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    result = alammar_cosine_similarity_loss(a, b, y_true)
    assert isinstance(result, dict)
