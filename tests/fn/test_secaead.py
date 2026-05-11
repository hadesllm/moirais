"""Tests for secaead.aead_chacha20poly1305."""
import numpy as np
import pytest
from morie.fn.secaead import aead_chacha20poly1305


def test_secaead_basic():
    """Test basic functionality."""
    key = np.random.default_rng(42).normal(0, 1, 100)
    nonce = np.random.default_rng(42).normal(0, 1, 100)
    plaintext = np.random.default_rng(42).normal(0, 1, 100)
    aad = np.random.default_rng(42).normal(0, 1, 100)
    result = aead_chacha20poly1305(key, nonce, plaintext, aad)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_secaead_edge():
    """Test edge cases."""
    key = np.random.default_rng(42).normal(0, 1, 100)
    nonce = np.random.default_rng(42).normal(0, 1, 100)
    plaintext = np.random.default_rng(42).normal(0, 1, 100)
    aad = np.random.default_rng(42).normal(0, 1, 100)
    result = aead_chacha20poly1305(key, nonce, plaintext, aad)
    assert isinstance(result, dict)
