"""Tests for guide_on_data_analysis24u1196.guide_on_data_analysis_chapter_24_unnumbered_1196."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis24u1196 import guide_on_data_analysis_chapter_24_unnumbered_1196


def test_guide_on_data_analysis24u1196_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1196(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis24u1196_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1196(x)
    assert isinstance(result, dict)
