"""Tests for smigr.smiles_grammar_parse."""
import numpy as np
import pytest
from moirais.fn.smigr import smiles_grammar_parse


def test_smigr_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = smiles_grammar_parse(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smigr_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = smiles_grammar_parse(smiles)
    assert isinstance(result, dict)
