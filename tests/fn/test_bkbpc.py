"""Tests for bkbpc.burkov_bits_per_character."""
import numpy as np
import pytest
from moirais.fn.bkbpc import burkov_bits_per_character


def test_bkbpc_basic():
    """Test basic functionality."""
    ce_loss = np.random.default_rng(42).normal(0, 1, 100)
    n_tokens = np.random.default_rng(42).normal(0, 1, 100)
    n_characters = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_bits_per_character(ce_loss, n_tokens, n_characters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkbpc_edge():
    """Test edge cases."""
    ce_loss = np.random.default_rng(42).normal(0, 1, 100)
    n_tokens = np.random.default_rng(42).normal(0, 1, 100)
    n_characters = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_bits_per_character(ce_loss, n_tokens, n_characters)
    assert isinstance(result, dict)
