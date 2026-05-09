"""Tests for guide_on_data_analysis24u1143.guide_on_data_analysis_chapter_24_unnumbered_1143."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis24u1143 import guide_on_data_analysis_chapter_24_unnumbered_1143


def test_guide_on_data_analysis24u1143_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1143(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis24u1143_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1143(x)
    assert isinstance(result, dict)
