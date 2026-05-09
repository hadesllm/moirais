"""Tests for rgcorart.rangayyan_coronary_sound."""
import numpy as np
import pytest
from moirais.fn.rgcorart import rangayyan_coronary_sound


def test_rgcorart_basic():
    """Test basic functionality."""
    diameter = np.random.default_rng(42).normal(0, 1, 100)
    flow_velocity = np.random.default_rng(42).normal(0, 1, 100)
    stenosis_pct = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_coronary_sound(diameter, flow_velocity, stenosis_pct)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcorart_edge():
    """Test edge cases."""
    diameter = np.random.default_rng(42).normal(0, 1, 100)
    flow_velocity = np.random.default_rng(42).normal(0, 1, 100)
    stenosis_pct = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_coronary_sound(diameter, flow_velocity, stenosis_pct)
    assert isinstance(result, dict)
