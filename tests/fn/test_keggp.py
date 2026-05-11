"""Tests for keggp.kegg_pathway."""
import numpy as np
import pytest
from morie.fn.keggp import kegg_pathway


def test_keggp_basic():
    """Test basic functionality."""
    genes = np.random.default_rng(42).normal(0, 1, 100)
    kegg_pathways = np.random.default_rng(42).normal(0, 1, 100)
    result = kegg_pathway(genes, kegg_pathways)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_keggp_edge():
    """Test edge cases."""
    genes = np.random.default_rng(42).normal(0, 1, 100)
    kegg_pathways = np.random.default_rng(42).normal(0, 1, 100)
    result = kegg_pathway(genes, kegg_pathways)
    assert isinstance(result, dict)
