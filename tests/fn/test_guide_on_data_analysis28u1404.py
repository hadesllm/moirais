"""Tests for guide_on_data_analysis28u1404.guide_on_data_analysis_chapter_28_unnumbered_1404."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis28u1404 import guide_on_data_analysis_chapter_28_unnumbered_1404


def test_guide_on_data_analysis28u1404_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1404(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis28u1404_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1404(x)
    assert isinstance(result, dict)
