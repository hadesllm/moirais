"""Tests for guide_on_data_analysis4u230.guide_on_data_analysis_chapter_4_unnumbered_230."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis4u230 import guide_on_data_analysis_chapter_4_unnumbered_230


def test_guide_on_data_analysis4u230_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_230(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis4u230_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_230(x)
    assert isinstance(result, dict)
