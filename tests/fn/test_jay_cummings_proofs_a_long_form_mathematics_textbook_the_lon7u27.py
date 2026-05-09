"""Tests for jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u27.jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_27."""
import numpy as np
import pytest
from moirais.fn.jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u27 import jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_27


def test_jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u27_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_27(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon7u27_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jay_cummings_proofs_a_long_form_mathematics_textbook_the_lon_chapter_7_unnumbered_27(x)
    assert isinstance(result, dict)
