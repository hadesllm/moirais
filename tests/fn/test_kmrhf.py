"""Tests for kmrhf.kamath_rlhf_pipeline."""
import numpy as np
import pytest
from moirais.fn.kmrhf import kamath_rlhf_pipeline


def test_kmrhf_basic():
    """Test basic functionality."""
    demos = np.random.default_rng(42).normal(0, 1, 100)
    preferences = np.random.default_rng(42).normal(0, 1, 100)
    pi0 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_rlhf_pipeline(demos, preferences, pi0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmrhf_edge():
    """Test edge cases."""
    demos = np.random.default_rng(42).normal(0, 1, 100)
    preferences = np.random.default_rng(42).normal(0, 1, 100)
    pi0 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_rlhf_pipeline(demos, preferences, pi0)
    assert isinstance(result, dict)
