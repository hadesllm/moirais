"""Tests for statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u173.statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_173."""
import numpy as np
import pytest
from moirais.fn.statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u173 import statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_173


def test_statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u173_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_173(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u173_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_173(x)
    assert isinstance(result, dict)
