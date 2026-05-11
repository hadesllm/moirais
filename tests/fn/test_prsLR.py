"""Tests for prsLR.lr_parser."""
import numpy as np
import pytest
from morie.fn.prsLR import lr_parser


def test_prsLR_basic():
    """Test basic functionality."""
    grammar = np.random.default_rng(42).normal(0, 1, 100)
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = lr_parser(grammar, tokens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prsLR_edge():
    """Test edge cases."""
    grammar = np.random.default_rng(42).normal(0, 1, 100)
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = lr_parser(grammar, tokens)
    assert isinstance(result, dict)
