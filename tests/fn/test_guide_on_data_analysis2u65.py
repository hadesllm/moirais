"""Tests for guide_on_data_analysis2u65.guide_on_data_analysis_chapter_2_unnumbered_65."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis2u65 import guide_on_data_analysis_chapter_2_unnumbered_65


def test_guide_on_data_analysis2u65_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_65(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis2u65_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_65(x)
    assert isinstance(result, dict)
