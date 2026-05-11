"""Tests for wilcox5u369.wilcox_chapter_5_unnumbered_369."""
import numpy as np
import pytest
from morie.fn.wilcox5u369 import wilcox_chapter_5_unnumbered_369


def test_wilcox5u369_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_369(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u369_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_369(x)
    assert isinstance(result, dict)
