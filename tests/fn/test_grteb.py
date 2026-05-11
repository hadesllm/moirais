"""Tests for grteb.geron_transformer_encoder_block."""
import numpy as np
import pytest
from morie.fn.grteb import geron_transformer_encoder_block


def test_grteb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mha_weights = np.random.default_rng(42).normal(0, 1, 100)
    ffn_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_transformer_encoder_block(x, mha_weights, ffn_weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grteb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mha_weights = np.random.default_rng(42).normal(0, 1, 100)
    ffn_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_transformer_encoder_block(x, mha_weights, ffn_weights)
    assert isinstance(result, dict)
