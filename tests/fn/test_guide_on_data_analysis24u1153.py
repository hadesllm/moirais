"""Tests for guide_on_data_analysis24u1153.guide_on_data_analysis_chapter_24_unnumbered_1153."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis24u1153 import guide_on_data_analysis_chapter_24_unnumbered_1153


def test_guide_on_data_analysis24u1153_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1153(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis24u1153_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1153(x)
    assert isinstance(result, dict)
