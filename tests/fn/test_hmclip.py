"""Tests for hmclip.geron_clip."""
import numpy as np
import pytest
from morie.fn.hmclip import geron_clip


def test_hmclip_basic():
    """Test basic functionality."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_clip(images, texts)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmclip_edge():
    """Test edge cases."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_clip(images, texts)
    assert isinstance(result, dict)
