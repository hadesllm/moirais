"""Tests for hmencd.geron_encoder_decoder_transformer."""
import numpy as np
import pytest
from morie.fn.hmencd import geron_encoder_decoder_transformer


def test_hmencd_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_decoder_transformer(src, tgt, n_layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmencd_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_decoder_transformer(src, tgt, n_layers)
    assert isinstance(result, dict)
