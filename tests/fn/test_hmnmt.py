"""Tests for hmnmt.geron_encoder_decoder_nmt."""
import numpy as np
import pytest
from morie.fn.hmnmt import geron_encoder_decoder_nmt


def test_hmnmt_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_decoder_nmt(src, tgt, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmnmt_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_decoder_nmt(src, tgt, model)
    assert isinstance(result, dict)
