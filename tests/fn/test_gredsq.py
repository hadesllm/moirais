"""Tests for gredsq.geron_encoder_decoder_seq2seq."""
import numpy as np
import pytest
from moirais.fn.gredsq import geron_encoder_decoder_seq2seq


def test_gredsq_basic():
    """Test basic functionality."""
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_out_len = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_decoder_seq2seq(encoder, decoder, x, max_out_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gredsq_edge():
    """Test edge cases."""
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_out_len = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_decoder_seq2seq(encoder, decoder, x, max_out_len)
    assert isinstance(result, dict)
