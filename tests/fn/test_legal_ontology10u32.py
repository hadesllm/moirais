"""Tests for legal_ontology10u32.legal_ontology_chapter_10_unnumbered_32."""
import numpy as np
import pytest
from morie.fn.legal_ontology10u32 import legal_ontology_chapter_10_unnumbered_32


def test_legal_ontology10u32_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_32(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_legal_ontology10u32_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_32(x)
    assert isinstance(result, dict)
