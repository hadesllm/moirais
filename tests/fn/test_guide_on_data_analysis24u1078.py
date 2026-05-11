"""Tests for guide_on_data_analysis24u1078.guide_on_data_analysis_chapter_24_unnumbered_1078."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis24u1078 import guide_on_data_analysis_chapter_24_unnumbered_1078


def test_guide_on_data_analysis24u1078_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1078(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis24u1078_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1078(x)
    assert isinstance(result, dict)
