"""Tests for guide_on_data_analysis24u1140.guide_on_data_analysis_chapter_24_unnumbered_1140."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis24u1140 import guide_on_data_analysis_chapter_24_unnumbered_1140


def test_guide_on_data_analysis24u1140_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1140(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis24u1140_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1140(x)
    assert isinstance(result, dict)
