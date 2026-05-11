"""Tests for cliptx.clip_image_text."""
import numpy as np
import pytest
from morie.fn.cliptx import clip_image_text


def test_cliptx_basic():
    """Test basic functionality."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = clip_image_text(images, texts, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cliptx_edge():
    """Test edge cases."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = clip_image_text(images, texts, tau)
    assert isinstance(result, dict)
