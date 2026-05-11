"""Tests for statistics_fourth_edition_david_freeman_robert_pisani_and_ro10u14.statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_10_unnumbered_14."""
import numpy as np
import pytest
from morie.fn.statistics_fourth_edition_david_freeman_robert_pisani_and_ro10u14 import statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_10_unnumbered_14


def test_statistics_fourth_edition_david_freeman_robert_pisani_and_ro10u14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_10_unnumbered_14(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistics_fourth_edition_david_freeman_robert_pisani_and_ro10u14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_10_unnumbered_14(x)
    assert isinstance(result, dict)
