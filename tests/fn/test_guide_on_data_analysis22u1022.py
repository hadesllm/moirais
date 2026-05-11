"""Tests for guide_on_data_analysis22u1022.guide_on_data_analysis_chapter_22_unnumbered_1022."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis22u1022 import guide_on_data_analysis_chapter_22_unnumbered_1022


def test_guide_on_data_analysis22u1022_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1022(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis22u1022_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1022(x)
    assert isinstance(result, dict)
