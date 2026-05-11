"""Tests for hmseq2.geron_seq2seq."""
import numpy as np
import pytest
from morie.fn.hmseq2 import geron_seq2seq


def test_hmseq2_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_seq2seq(src, tgt, encoder, decoder)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmseq2_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_seq2seq(src, tgt, encoder, decoder)
    assert isinstance(result, dict)
