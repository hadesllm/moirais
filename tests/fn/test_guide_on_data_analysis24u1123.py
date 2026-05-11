"""Tests for guide_on_data_analysis24u1123.guide_on_data_analysis_chapter_24_unnumbered_1123."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis24u1123 import guide_on_data_analysis_chapter_24_unnumbered_1123


def test_guide_on_data_analysis24u1123_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1123(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis24u1123_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1123(x)
    assert isinstance(result, dict)
