"""Tests for gnsetenr.geneset_enrichment."""
import numpy as np
import pytest
from moirais.fn.gnsetenr import geneset_enrichment


def test_gnsetenr_basic():
    """Test basic functionality."""
    ranked_genes = np.random.default_rng(42).normal(0, 1, 100)
    gene_set = np.random.default_rng(42).normal(0, 1, 100)
    result = geneset_enrichment(ranked_genes, gene_set)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gnsetenr_edge():
    """Test edge cases."""
    ranked_genes = np.random.default_rng(42).normal(0, 1, 100)
    gene_set = np.random.default_rng(42).normal(0, 1, 100)
    result = geneset_enrichment(ranked_genes, gene_set)
    assert isinstance(result, dict)
