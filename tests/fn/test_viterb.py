"""Tests for viterb.viterbi."""
import numpy as np
import pytest
from moirais.fn.viterb import viterbi


def test_viterb_basic():
    """Test basic functionality."""
    obs = np.random.default_rng(42).normal(0, 1, 100)
    trans = np.random.default_rng(42).normal(0, 1, 100)
    emit = np.random.default_rng(42).normal(0, 1, 100)
    result = viterbi(obs, trans, emit)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_viterb_edge():
    """Test edge cases."""
    obs = np.random.default_rng(42).normal(0, 1, 100)
    trans = np.random.default_rng(42).normal(0, 1, 100)
    emit = np.random.default_rng(42).normal(0, 1, 100)
    result = viterbi(obs, trans, emit)
    assert isinstance(result, dict)
