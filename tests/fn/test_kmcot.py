"""Tests for kmcot.kamath_chain_of_thought."""
import numpy as np
import pytest
from moirais.fn.kmcot import kamath_chain_of_thought


def test_kmcot_basic():
    """Test basic functionality."""
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_chain_of_thought(prompt, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmcot_edge():
    """Test edge cases."""
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_chain_of_thought(prompt, model)
    assert isinstance(result, dict)
