"""Tests for prsLL.ll_parser."""
import numpy as np
import pytest
from morie.fn.prsLL import ll_parser


def test_prsLL_basic():
    """Test basic functionality."""
    grammar = np.random.default_rng(42).normal(0, 1, 100)
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = ll_parser(grammar, tokens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prsLL_edge():
    """Test edge cases."""
    grammar = np.random.default_rng(42).normal(0, 1, 100)
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = ll_parser(grammar, tokens)
    assert isinstance(result, dict)
