"""Tests for clipsi.clip_similarity."""
import numpy as np
import pytest
from moirais.fn.clipsi import clip_similarity


def test_clipsi_basic():
    """Test basic functionality."""
    I_emb = np.random.default_rng(42).normal(0, 1, 100)
    T_emb = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = clip_similarity(I_emb, T_emb, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clipsi_edge():
    """Test edge cases."""
    I_emb = np.random.default_rng(42).normal(0, 1, 100)
    T_emb = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = clip_similarity(I_emb, T_emb, tau)
    assert isinstance(result, dict)
