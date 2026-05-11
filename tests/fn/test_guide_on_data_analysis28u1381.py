"""Tests for guide_on_data_analysis28u1381.guide_on_data_analysis_chapter_28_unnumbered_1381."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis28u1381 import guide_on_data_analysis_chapter_28_unnumbered_1381


def test_guide_on_data_analysis28u1381_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1381(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis28u1381_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1381(x)
    assert isinstance(result, dict)
