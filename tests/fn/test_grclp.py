"""Tests for grclp.geron_clip_contrastive_loss."""
import numpy as np
import pytest
from moirais.fn.grclp import geron_clip_contrastive_loss


def test_grclp_basic():
    """Test basic functionality."""
    image_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    text_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = geron_clip_contrastive_loss(image_embeddings, text_embeddings, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grclp_edge():
    """Test edge cases."""
    image_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    text_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = geron_clip_contrastive_loss(image_embeddings, text_embeddings, tau)
    assert isinstance(result, dict)
