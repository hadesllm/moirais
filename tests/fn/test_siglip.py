"""Tests for siglip.siglip_pairwise."""
import numpy as np
import pytest
from morie.fn.siglip import siglip_pairwise


def test_siglip_basic():
    """Test basic functionality."""
    image_emb = np.random.default_rng(42).normal(0, 1, 100)
    text_emb = np.random.default_rng(42).normal(0, 1, 100)
    result = siglip_pairwise(image_emb, text_emb)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_siglip_edge():
    """Test edge cases."""
    image_emb = np.random.default_rng(42).normal(0, 1, 100)
    text_emb = np.random.default_rng(42).normal(0, 1, 100)
    result = siglip_pairwise(image_emb, text_emb)
    assert isinstance(result, dict)
