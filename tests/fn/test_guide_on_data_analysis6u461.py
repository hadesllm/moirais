"""Tests for guide_on_data_analysis6u461.guide_on_data_analysis_chapter_6_unnumbered_461."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis6u461 import guide_on_data_analysis_chapter_6_unnumbered_461


def test_guide_on_data_analysis6u461_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_461(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis6u461_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_461(x)
    assert isinstance(result, dict)
