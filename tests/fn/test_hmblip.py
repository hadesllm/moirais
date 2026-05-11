"""Tests for hmblip.geron_blip."""
import numpy as np
import pytest
from morie.fn.hmblip import geron_blip


def test_hmblip_basic():
    """Test basic functionality."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_blip(images, texts)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmblip_edge():
    """Test edge cases."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_blip(images, texts)
    assert isinstance(result, dict)
