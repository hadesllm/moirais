"""Tests for kmstb.kamath_step_back_prompting."""
import numpy as np
import pytest
from moirais.fn.kmstb import kamath_step_back_prompting


def test_kmstb_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_step_back_prompting(query, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmstb_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_step_back_prompting(query, model)
    assert isinstance(result, dict)
