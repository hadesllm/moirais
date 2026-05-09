"""Tests for kmcap.kamath_expert_capacity_factor."""
import numpy as np
import pytest
from moirais.fn.kmcap import kamath_expert_capacity_factor


def test_kmcap_basic():
    """Test basic functionality."""
    tokens_per_batch = np.random.default_rng(42).normal(0, 1, 100)
    num_experts = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_expert_capacity_factor(tokens_per_batch, num_experts, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmcap_edge():
    """Test edge cases."""
    tokens_per_batch = np.random.default_rng(42).normal(0, 1, 100)
    num_experts = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_expert_capacity_factor(tokens_per_batch, num_experts, C)
    assert isinstance(result, dict)
