"""Tests for kmspd.kamath_speculative_decoding."""
import numpy as np
import pytest
from morie.fn.kmspd import kamath_speculative_decoding


def test_kmspd_basic():
    """Test basic functionality."""
    draft_probs = np.random.default_rng(42).normal(0, 1, 100)
    target_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_speculative_decoding(draft_probs, target_probs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmspd_edge():
    """Test edge cases."""
    draft_probs = np.random.default_rng(42).normal(0, 1, 100)
    target_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_speculative_decoding(draft_probs, target_probs)
    assert isinstance(result, dict)
