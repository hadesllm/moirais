"""Tests for vitcls.vit_cls_token."""
import numpy as np
import pytest
from morie.fn.vitcls import vit_cls_token


def test_vitcls_basic():
    """Test basic functionality."""
    patches = np.random.default_rng(42).normal(0, 1, 100)
    n_patches = np.random.default_rng(42).normal(0, 1, 100)
    result = vit_cls_token(patches, n_patches)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vitcls_edge():
    """Test edge cases."""
    patches = np.random.default_rng(42).normal(0, 1, 100)
    n_patches = np.random.default_rng(42).normal(0, 1, 100)
    result = vit_cls_token(patches, n_patches)
    assert isinstance(result, dict)
