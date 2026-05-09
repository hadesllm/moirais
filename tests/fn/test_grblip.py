"""Tests for grblip.geron_blip_itm_itc."""
import numpy as np
import pytest
from moirais.fn.grblip import geron_blip_itm_itc


def test_grblip_basic():
    """Test basic functionality."""
    image_emb = np.random.default_rng(42).normal(0, 1, 100)
    text_emb = np.random.default_rng(42).normal(0, 1, 100)
    caption_logits = np.random.default_rng(42).normal(0, 1, 100)
    caption_targets = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_blip_itm_itc(image_emb, text_emb, caption_logits, caption_targets)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grblip_edge():
    """Test edge cases."""
    image_emb = np.random.default_rng(42).normal(0, 1, 100)
    text_emb = np.random.default_rng(42).normal(0, 1, 100)
    caption_logits = np.random.default_rng(42).normal(0, 1, 100)
    caption_targets = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_blip_itm_itc(image_emb, text_emb, caption_logits, caption_targets)
    assert isinstance(result, dict)
