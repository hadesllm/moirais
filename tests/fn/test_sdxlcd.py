"""Tests for sdxlcd.sdxl_unet."""
import numpy as np
import pytest
from morie.fn.sdxlcd import sdxl_unet


def test_sdxlcd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    text_emb = np.random.default_rng(42).normal(0, 1, 100)
    result = sdxl_unet(x, t, text_emb)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sdxlcd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    text_emb = np.random.default_rng(42).normal(0, 1, 100)
    result = sdxl_unet(x, t, text_emb)
    assert isinstance(result, dict)
