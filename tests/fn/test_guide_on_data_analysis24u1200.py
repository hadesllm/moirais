"""Tests for guide_on_data_analysis24u1200.guide_on_data_analysis_chapter_24_unnumbered_1200."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis24u1200 import guide_on_data_analysis_chapter_24_unnumbered_1200


def test_guide_on_data_analysis24u1200_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1200(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis24u1200_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1200(x)
    assert isinstance(result, dict)
