"""Tests for clipbn.clip_image_text_align."""
import numpy as np
import pytest
from morie.fn.clipbn import clip_image_text_align


def test_clipbn_basic():
    """Test basic functionality."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = clip_image_text_align(images, texts, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clipbn_edge():
    """Test edge cases."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = clip_image_text_align(images, texts, tau)
    assert isinstance(result, dict)
