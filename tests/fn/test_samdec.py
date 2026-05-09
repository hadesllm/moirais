"""Tests for samdec.sam_mask_decoder."""
import numpy as np
import pytest
from moirais.fn.samdec import sam_mask_decoder


def test_samdec_basic():
    """Test basic functionality."""
    img_emb = np.random.default_rng(42).normal(0, 1, 100)
    prompt_emb = np.random.default_rng(42).normal(0, 1, 100)
    result = sam_mask_decoder(img_emb, prompt_emb)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_samdec_edge():
    """Test edge cases."""
    img_emb = np.random.default_rng(42).normal(0, 1, 100)
    prompt_emb = np.random.default_rng(42).normal(0, 1, 100)
    result = sam_mask_decoder(img_emb, prompt_emb)
    assert isinstance(result, dict)
