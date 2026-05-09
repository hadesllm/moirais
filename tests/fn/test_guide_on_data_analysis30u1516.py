"""Tests for guide_on_data_analysis30u1516.guide_on_data_analysis_chapter_30_unnumbered_1516."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis30u1516 import guide_on_data_analysis_chapter_30_unnumbered_1516


def test_guide_on_data_analysis30u1516_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1516(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis30u1516_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1516(x)
    assert isinstance(result, dict)
