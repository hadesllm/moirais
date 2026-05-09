"""Tests for vqgenc.vqgan_encode."""
import numpy as np
import pytest
from moirais.fn.vqgenc import vqgan_encode


def test_vqgenc_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    codebook = np.random.default_rng(42).normal(0, 1, 100)
    result = vqgan_encode(image, codebook)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vqgenc_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    codebook = np.random.default_rng(42).normal(0, 1, 100)
    result = vqgan_encode(image, codebook)
    assert isinstance(result, dict)
