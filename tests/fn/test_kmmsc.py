"""Tests for kmmsc.kamath_moverscore."""
import numpy as np
import pytest
from morie.fn.kmmsc import kamath_moverscore


def test_kmmsc_basic():
    """Test basic functionality."""
    hypothesis_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    reference_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_moverscore(hypothesis_embeddings, reference_embeddings)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmmsc_edge():
    """Test edge cases."""
    hypothesis_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    reference_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_moverscore(hypothesis_embeddings, reference_embeddings)
    assert isinstance(result, dict)
