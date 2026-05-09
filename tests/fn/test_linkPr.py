"""Tests for linkPr.link_prediction."""
import numpy as np
import pytest
from moirais.fn.linkPr import link_prediction


def test_linkPr_basic():
    """Test basic functionality."""
    G = np.eye(10)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    method = 'auto'
    result = link_prediction(G, u, v, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_linkPr_edge():
    """Test edge cases."""
    G = np.eye(10)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    method = 'auto'
    result = link_prediction(G, u, v, method)
    assert isinstance(result, dict)
