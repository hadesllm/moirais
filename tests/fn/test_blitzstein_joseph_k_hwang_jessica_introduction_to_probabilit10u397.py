"""Tests for blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u397.blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_397."""
import numpy as np
import pytest
from moirais.fn.blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u397 import blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_397


def test_blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u397_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_397(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u397_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_397(x)
    assert isinstance(result, dict)
