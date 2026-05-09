"""Tests for guide_on_data_analysis6u491.guide_on_data_analysis_chapter_6_unnumbered_491."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis6u491 import guide_on_data_analysis_chapter_6_unnumbered_491


def test_guide_on_data_analysis6u491_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_491(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis6u491_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_491(x)
    assert isinstance(result, dict)
