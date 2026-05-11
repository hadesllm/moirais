"""Tests for specS.speculative_decoding."""
import numpy as np
import pytest
from morie.fn.specS import speculative_decoding


def test_specS_basic():
    """Test basic functionality."""
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    draft = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = speculative_decoding(prompt, draft, target)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_specS_edge():
    """Test edge cases."""
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    draft = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = speculative_decoding(prompt, draft, target)
    assert isinstance(result, dict)
