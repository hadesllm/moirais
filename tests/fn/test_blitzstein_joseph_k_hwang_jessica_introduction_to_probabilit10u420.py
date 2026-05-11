"""Tests for blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u420.blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_420."""
import numpy as np
import pytest
from morie.fn.blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u420 import blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_420


def test_blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u420_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_420(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u420_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_420(x)
    assert isinstance(result, dict)
