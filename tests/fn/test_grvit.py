"""Tests for grvit.geron_vit_patch_embedding."""
import numpy as np
import pytest
from moirais.fn.grvit import geron_vit_patch_embedding


def test_grvit_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    E = np.random.default_rng(42).normal(0, 1, 100)
    E_pos = np.random.default_rng(42).normal(0, 1, 100)
    cls_token = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vit_patch_embedding(image, patch_size, E, E_pos, cls_token)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grvit_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    E = np.random.default_rng(42).normal(0, 1, 100)
    E_pos = np.random.default_rng(42).normal(0, 1, 100)
    cls_token = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vit_patch_embedding(image, patch_size, E, E_pos, cls_token)
    assert isinstance(result, dict)
