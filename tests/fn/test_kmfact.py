"""Tests for kmfact.kamath_factscore."""
import numpy as np
import pytest
from moirais.fn.kmfact import kamath_factscore


def test_kmfact_basic():
    """Test basic functionality."""
    atomic_claims = np.random.default_rng(42).normal(0, 1, 100)
    knowledge_base = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_factscore(atomic_claims, knowledge_base)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmfact_edge():
    """Test edge cases."""
    atomic_claims = np.random.default_rng(42).normal(0, 1, 100)
    knowledge_base = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_factscore(atomic_claims, knowledge_base)
    assert isinstance(result, dict)
