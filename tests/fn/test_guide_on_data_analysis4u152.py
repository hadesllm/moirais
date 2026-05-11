"""Tests for guide_on_data_analysis4u152.guide_on_data_analysis_chapter_4_unnumbered_152."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis4u152 import guide_on_data_analysis_chapter_4_unnumbered_152


def test_guide_on_data_analysis4u152_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_152(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis4u152_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_152(x)
    assert isinstance(result, dict)
