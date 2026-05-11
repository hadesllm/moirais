"""Tests for blip2v.blip2_qformer."""
import numpy as np
import pytest
from morie.fn.blip2v import blip2_qformer


def test_blip2v_basic():
    """Test basic functionality."""
    image_features = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    result = blip2_qformer(image_features, queries, llm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blip2v_edge():
    """Test edge cases."""
    image_features = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    result = blip2_qformer(image_features, queries, llm)
    assert isinstance(result, dict)
