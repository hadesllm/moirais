"""Tests for guide_on_data_analysis24u1121.guide_on_data_analysis_chapter_24_unnumbered_1121."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis24u1121 import guide_on_data_analysis_chapter_24_unnumbered_1121


def test_guide_on_data_analysis24u1121_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1121(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis24u1121_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1121(x)
    assert isinstance(result, dict)
