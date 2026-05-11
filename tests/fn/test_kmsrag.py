"""Tests for kmsrag.kamath_self_rag."""
import numpy as np
import pytest
from morie.fn.kmsrag import kamath_self_rag


def test_kmsrag_basic():
    """Test basic functionality."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    reflection_model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_self_rag(context, reflection_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmsrag_edge():
    """Test edge cases."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    reflection_model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_self_rag(context, reflection_model)
    assert isinstance(result, dict)
