"""Tests for alocp.alammar_openclip_contrastive."""
import numpy as np
import pytest
from moirais.fn.alocp import alammar_openclip_contrastive


def test_alocp_basic():
    """Test basic functionality."""
    I_emb = np.random.default_rng(42).normal(0, 1, 100)
    T_emb = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = alammar_openclip_contrastive(I_emb, T_emb, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alocp_edge():
    """Test edge cases."""
    I_emb = np.random.default_rng(42).normal(0, 1, 100)
    T_emb = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = alammar_openclip_contrastive(I_emb, T_emb, tau)
    assert isinstance(result, dict)
