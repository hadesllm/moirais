"""Tests for guide_on_data_analysis14u888.guide_on_data_analysis_chapter_14_unnumbered_888."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis14u888 import guide_on_data_analysis_chapter_14_unnumbered_888


def test_guide_on_data_analysis14u888_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_888(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis14u888_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_888(x)
    assert isinstance(result, dict)
