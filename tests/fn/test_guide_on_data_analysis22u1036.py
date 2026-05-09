"""Tests for guide_on_data_analysis22u1036.guide_on_data_analysis_chapter_22_unnumbered_1036."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis22u1036 import guide_on_data_analysis_chapter_22_unnumbered_1036


def test_guide_on_data_analysis22u1036_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1036(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis22u1036_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1036(x)
    assert isinstance(result, dict)
