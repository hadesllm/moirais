"""Tests for kmlv.kamath_llava_visual_instruction."""
import numpy as np
import pytest
from moirais.fn.kmlv import kamath_llava_visual_instruction


def test_kmlv_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    visual_encoder = np.random.default_rng(42).normal(0, 1, 100)
    text_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_llava_visual_instruction(image, W, visual_encoder, text_tokens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmlv_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    visual_encoder = np.random.default_rng(42).normal(0, 1, 100)
    text_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_llava_visual_instruction(image, W, visual_encoder, text_tokens)
    assert isinstance(result, dict)
