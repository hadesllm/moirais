"""Tests for hmnpl.geron_neurons_per_layer."""
import numpy as np
import pytest
from moirais.fn.hmnpl import geron_neurons_per_layer


def test_hmnpl_basic():
    """Test basic functionality."""
    n_features = 5
    result = geron_neurons_per_layer(n_features)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmnpl_edge():
    """Test edge cases."""
    n_features = 5
    result = geron_neurons_per_layer(n_features)
    assert isinstance(result, dict)
