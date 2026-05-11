"""Tests for vitptm.vit_patch_embed."""
import numpy as np
import pytest
from morie.fn.vitptm import vit_patch_embed


def test_vitptm_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    embed_dim = 2
    result = vit_patch_embed(image, patch_size, embed_dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vitptm_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    embed_dim = 2
    result = vit_patch_embed(image, patch_size, embed_dim)
    assert isinstance(result, dict)
