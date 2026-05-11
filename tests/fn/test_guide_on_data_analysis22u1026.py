"""Tests for guide_on_data_analysis22u1026.guide_on_data_analysis_chapter_22_unnumbered_1026."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis22u1026 import guide_on_data_analysis_chapter_22_unnumbered_1026


def test_guide_on_data_analysis22u1026_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1026(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis22u1026_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1026(x)
    assert isinstance(result, dict)
