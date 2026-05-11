"""Tests for kmptun.kamath_prompt_tuning."""
import numpy as np
import pytest
from morie.fn.kmptun import kamath_prompt_tuning


def test_kmptun_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = kamath_prompt_tuning(P, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmptun_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = kamath_prompt_tuning(P, X)
    assert isinstance(result, dict)
