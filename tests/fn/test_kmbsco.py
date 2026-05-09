"""Tests for kmbsco.kamath_bertscore."""
import numpy as np
import pytest
from moirais.fn.kmbsco import kamath_bertscore


def test_kmbsco_basic():
    """Test basic functionality."""
    hypothesis_tokens = np.random.default_rng(42).normal(0, 1, 100)
    reference_tokens = np.random.default_rng(42).normal(0, 1, 100)
    embed_fn = (lambda v: v)
    result = kamath_bertscore(hypothesis_tokens, reference_tokens, embed_fn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmbsco_edge():
    """Test edge cases."""
    hypothesis_tokens = np.random.default_rng(42).normal(0, 1, 100)
    reference_tokens = np.random.default_rng(42).normal(0, 1, 100)
    embed_fn = (lambda v: v)
    result = kamath_bertscore(hypothesis_tokens, reference_tokens, embed_fn)
    assert isinstance(result, dict)
