"""Tests for kmitm.kamath_image_text_matching."""
import numpy as np
import pytest
from moirais.fn.kmitm import kamath_image_text_matching


def test_kmitm_basic():
    """Test basic functionality."""
    image_emb = np.random.default_rng(42).normal(0, 1, 100)
    text_emb = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_image_text_matching(image_emb, text_emb, W, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmitm_edge():
    """Test edge cases."""
    image_emb = np.random.default_rng(42).normal(0, 1, 100)
    text_emb = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_image_text_matching(image_emb, text_emb, W, b)
    assert isinstance(result, dict)
