"""Tests for kmpoln.kamath_post_ln_transformer."""
import numpy as np
import pytest
from morie.fn.kmpoln import kamath_post_ln_transformer


def test_kmpoln_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    attn_fn = (lambda v: v)
    ffn_fn = (lambda v: v)
    result = kamath_post_ln_transformer(x, attn_fn, ffn_fn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmpoln_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    attn_fn = (lambda v: v)
    ffn_fn = (lambda v: v)
    result = kamath_post_ln_transformer(x, attn_fn, ffn_fn)
    assert isinstance(result, dict)
