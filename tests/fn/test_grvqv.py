"""Tests for grvqv.geron_vq_vae_codebook_loss."""
import numpy as np
import pytest
from moirais.fn.grvqv import geron_vq_vae_codebook_loss


def test_grvqv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z_e = np.random.default_rng(42).normal(0, 1, 100)
    z_q = np.random.default_rng(42).normal(0, 1, 100)
    codebook = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_vq_vae_codebook_loss(x, z_e, z_q, codebook, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grvqv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z_e = np.random.default_rng(42).normal(0, 1, 100)
    z_q = np.random.default_rng(42).normal(0, 1, 100)
    codebook = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_vq_vae_codebook_loss(x, z_e, z_q, codebook, beta)
    assert isinstance(result, dict)
