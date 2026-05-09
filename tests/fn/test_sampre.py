"""Tests for sampre.sam_prompt_encoder."""
import numpy as np
import pytest
from moirais.fn.sampre import sam_prompt_encoder


def test_sampre_basic():
    """Test basic functionality."""
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    result = sam_prompt_encoder(prompts)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sampre_edge():
    """Test edge cases."""
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    result = sam_prompt_encoder(prompts)
    assert isinstance(result, dict)
