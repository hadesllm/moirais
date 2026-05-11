"""Tests for kmbpb.kamath_bits_per_byte."""
import numpy as np
import pytest
from morie.fn.kmbpb import kamath_bits_per_byte


def test_kmbpb_basic():
    """Test basic functionality."""
    ce_loss = np.random.default_rng(42).normal(0, 1, 100)
    n_tokens = np.random.default_rng(42).normal(0, 1, 100)
    n_bytes = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_bits_per_byte(ce_loss, n_tokens, n_bytes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmbpb_edge():
    """Test edge cases."""
    ce_loss = np.random.default_rng(42).normal(0, 1, 100)
    n_tokens = np.random.default_rng(42).normal(0, 1, 100)
    n_bytes = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_bits_per_byte(ce_loss, n_tokens, n_bytes)
    assert isinstance(result, dict)
