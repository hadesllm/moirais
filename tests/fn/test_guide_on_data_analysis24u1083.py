"""Tests for guide_on_data_analysis24u1083.guide_on_data_analysis_chapter_24_unnumbered_1083."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis24u1083 import guide_on_data_analysis_chapter_24_unnumbered_1083


def test_guide_on_data_analysis24u1083_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1083(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis24u1083_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1083(x)
    assert isinstance(result, dict)
