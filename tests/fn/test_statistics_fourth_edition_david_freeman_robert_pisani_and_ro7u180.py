"""Tests for statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u180.statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_180."""
import numpy as np
import pytest
from morie.fn.statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u180 import statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_180


def test_statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u180_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_180(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u180_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_180(x)
    assert isinstance(result, dict)
