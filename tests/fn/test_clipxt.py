"""Tests for clipxt.clip_text_encoder."""
import numpy as np
import pytest
from moirais.fn.clipxt import clip_text_encoder


def test_clipxt_basic():
    """Test basic functionality."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    result = clip_text_encoder(text)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clipxt_edge():
    """Test edge cases."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    result = clip_text_encoder(text)
    assert isinstance(result, dict)
