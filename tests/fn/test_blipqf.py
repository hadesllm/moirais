"""Tests for blipqf.blip_qformer."""
import numpy as np
import pytest
from moirais.fn.blipqf import blip_qformer


def test_blipqf_basic():
    """Test basic functionality."""
    image_features = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    result = blip_qformer(image_features, queries, llm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blipqf_edge():
    """Test edge cases."""
    image_features = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    result = blip_qformer(image_features, queries, llm)
    assert isinstance(result, dict)
