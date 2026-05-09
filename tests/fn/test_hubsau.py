"""He who is brave is free. — Seneca"""
import numpy as np
import pytest
from moirais.fn.hubsau import hits_hub_authority


def test_hubsau_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = hits_hub_authority(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hubsau_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = hits_hub_authority(G)
    assert isinstance(result, dict)
