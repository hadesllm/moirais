"""Tests for vqgdec.vqgan_decode."""
import numpy as np
import pytest
from moirais.fn.vqgdec import vqgan_decode


def test_vqgdec_basic():
    """Test basic functionality."""
    indices = np.arange(0, 100, dtype=int)
    codebook = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = vqgan_decode(indices, codebook, decoder)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vqgdec_edge():
    """Test edge cases."""
    indices = np.arange(0, 100, dtype=int)
    codebook = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = vqgan_decode(indices, codebook, decoder)
    assert isinstance(result, dict)
