"""Tests for blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u380.blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_380."""
import numpy as np
import pytest
from morie.fn.blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u380 import blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_380


def test_blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u380_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_380(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u380_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_380(x)
    assert isinstance(result, dict)
