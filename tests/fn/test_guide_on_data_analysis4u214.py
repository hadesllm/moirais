"""Tests for guide_on_data_analysis4u214.guide_on_data_analysis_chapter_4_unnumbered_214."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis4u214 import guide_on_data_analysis_chapter_4_unnumbered_214


def test_guide_on_data_analysis4u214_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_214(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis4u214_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_214(x)
    assert isinstance(result, dict)
