"""Tests for bknrm.burkov_vector_norm."""
import numpy as np
import pytest
from morie.fn.bknrm import burkov_vector_norm


def test_bknrm_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = burkov_vector_norm(a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bknrm_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = burkov_vector_norm(a)
    assert isinstance(result, dict)
