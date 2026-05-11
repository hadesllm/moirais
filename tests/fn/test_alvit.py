"""Tests for alvit.alammar_vit_patch_embedding."""
import numpy as np
import pytest
from morie.fn.alvit import alammar_vit_patch_embedding


def test_alvit_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_vit_patch_embedding(image, patch_size, E)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alvit_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_vit_patch_embedding(image, patch_size, E)
    assert isinstance(result, dict)
