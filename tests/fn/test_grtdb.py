"""Tests for grtdb.geron_transformer_decoder_block."""
import numpy as np
import pytest
from morie.fn.grtdb import geron_transformer_decoder_block


def test_grtdb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    encoder_output = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_transformer_decoder_block(x, encoder_output, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grtdb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    encoder_output = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_transformer_decoder_block(x, encoder_output, weights)
    assert isinstance(result, dict)
