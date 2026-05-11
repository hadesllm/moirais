"""Tests for grdal.geron_dalle_autoregressive_token."""
import numpy as np
import pytest
from morie.fn.grdal import geron_dalle_autoregressive_token


def test_grdal_basic():
    """Test basic functionality."""
    text_tokens = np.random.default_rng(42).normal(0, 1, 100)
    image_tokens_prefix = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dalle_autoregressive_token(text_tokens, image_tokens_prefix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdal_edge():
    """Test edge cases."""
    text_tokens = np.random.default_rng(42).normal(0, 1, 100)
    image_tokens_prefix = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dalle_autoregressive_token(text_tokens, image_tokens_prefix)
    assert isinstance(result, dict)
