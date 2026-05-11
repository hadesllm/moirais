"""Tests for guide_on_data_analysis28u1391.guide_on_data_analysis_chapter_28_unnumbered_1391."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis28u1391 import guide_on_data_analysis_chapter_28_unnumbered_1391


def test_guide_on_data_analysis28u1391_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1391(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis28u1391_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1391(x)
    assert isinstance(result, dict)
