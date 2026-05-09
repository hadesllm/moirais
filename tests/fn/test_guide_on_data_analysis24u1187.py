"""Tests for guide_on_data_analysis24u1187.guide_on_data_analysis_chapter_24_unnumbered_1187."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis24u1187 import guide_on_data_analysis_chapter_24_unnumbered_1187


def test_guide_on_data_analysis24u1187_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1187(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis24u1187_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1187(x)
    assert isinstance(result, dict)
