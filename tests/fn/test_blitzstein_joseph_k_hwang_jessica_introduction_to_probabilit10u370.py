"""Tests for blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u370.blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_370."""
import numpy as np
import pytest
from morie.fn.blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u370 import blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_370


def test_blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u370_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_370(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u370_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_370(x)
    assert isinstance(result, dict)
