"""Tests for jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u25.jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_25."""
import numpy as np
import pytest
from moirais.fn.jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u25 import jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_25


def test_jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u25_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_25(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u25_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_25(x)
    assert isinstance(result, dict)
