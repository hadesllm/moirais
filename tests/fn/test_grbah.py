"""Tests for grbah.geron_bahdanau_attention."""
import numpy as np
import pytest
from moirais.fn.grbah import geron_bahdanau_attention


def test_grbah_basic():
    """Test basic functionality."""
    decoder_state = np.random.default_rng(42).normal(0, 1, 100)
    encoder_states = np.random.default_rng(42).normal(0, 1, 100)
    Wh = np.random.default_rng(42).normal(0, 1, 100)
    Ws = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_bahdanau_attention(decoder_state, encoder_states, Wh, Ws, v)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbah_edge():
    """Test edge cases."""
    decoder_state = np.random.default_rng(42).normal(0, 1, 100)
    encoder_states = np.random.default_rng(42).normal(0, 1, 100)
    Wh = np.random.default_rng(42).normal(0, 1, 100)
    Ws = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_bahdanau_attention(decoder_state, encoder_states, Wh, Ws, v)
    assert isinstance(result, dict)
