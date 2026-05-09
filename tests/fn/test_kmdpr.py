"""Tests for kmdpr.kamath_dense_passage_retrieval."""
import numpy as np
import pytest
from moirais.fn.kmdpr import kamath_dense_passage_retrieval


def test_kmdpr_basic():
    """Test basic functionality."""
    q_embed = np.random.default_rng(42).normal(0, 1, 100)
    p_embeds = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_dense_passage_retrieval(q_embed, p_embeds, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmdpr_edge():
    """Test edge cases."""
    q_embed = np.random.default_rng(42).normal(0, 1, 100)
    p_embeds = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_dense_passage_retrieval(q_embed, p_embeds, k)
    assert isinstance(result, dict)
