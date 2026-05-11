"""Tests for kmcai.kamath_constitutional_ai_loop."""
import numpy as np
import pytest
from morie.fn.kmcai import kamath_constitutional_ai_loop


def test_kmcai_basic():
    """Test basic functionality."""
    initial_response = np.random.default_rng(42).normal(0, 1, 100)
    constitution = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_constitutional_ai_loop(initial_response, constitution, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmcai_edge():
    """Test edge cases."""
    initial_response = np.random.default_rng(42).normal(0, 1, 100)
    constitution = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_constitutional_ai_loop(initial_response, constitution, model)
    assert isinstance(result, dict)
