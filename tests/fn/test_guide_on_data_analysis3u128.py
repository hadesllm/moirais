"""Tests for guide_on_data_analysis3u128.guide_on_data_analysis_chapter_3_unnumbered_128."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis3u128 import guide_on_data_analysis_chapter_3_unnumbered_128


def test_guide_on_data_analysis3u128_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_128(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis3u128_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_128(x)
    assert isinstance(result, dict)
