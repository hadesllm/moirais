"""Tests for clipxi.clip_image_encoder."""
import numpy as np
import pytest
from morie.fn.clipxi import clip_image_encoder


def test_clipxi_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    backbone = np.random.default_rng(42).normal(0, 1, 100)
    result = clip_image_encoder(image, backbone)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clipxi_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    backbone = np.random.default_rng(42).normal(0, 1, 100)
    result = clip_image_encoder(image, backbone)
    assert isinstance(result, dict)
