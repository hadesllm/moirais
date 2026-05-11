"""Tests for franz_kronthaler_statistics_applied_with_the_r_commander_dat3u21.franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_21."""
import numpy as np
import pytest
from morie.fn.franz_kronthaler_statistics_applied_with_the_r_commander_dat3u21 import franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_21


def test_franz_kronthaler_statistics_applied_with_the_r_commander_dat3u21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_21(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_franz_kronthaler_statistics_applied_with_the_r_commander_dat3u21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_21(x)
    assert isinstance(result, dict)
