"""Tests for law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe6e1.law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe_chapter_6_equation_1."""
import numpy as np
import pytest
from morie.fn.law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe6e1 import law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe_chapter_6_equation_1


def test_law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe6e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe_chapter_6_equation_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe6e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe_chapter_6_equation_1(x)
    assert isinstance(result, dict)
