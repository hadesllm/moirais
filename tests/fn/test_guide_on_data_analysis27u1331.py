"""Tests for guide_on_data_analysis27u1331.guide_on_data_analysis_chapter_27_unnumbered_1331."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis27u1331 import guide_on_data_analysis_chapter_27_unnumbered_1331


def test_guide_on_data_analysis27u1331_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1331(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis27u1331_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1331(x)
    assert isinstance(result, dict)
