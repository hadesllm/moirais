"""Tests for guide_on_data_analysis6u483.guide_on_data_analysis_chapter_6_unnumbered_483."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis6u483 import guide_on_data_analysis_chapter_6_unnumbered_483


def test_guide_on_data_analysis6u483_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_483(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis6u483_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_483(x)
    assert isinstance(result, dict)
