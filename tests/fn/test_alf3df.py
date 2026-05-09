"""Tests for alf3df.af3_diffusion_step."""
import numpy as np
import pytest
from moirais.fn.alf3df import af3_diffusion_step


def test_alf3df_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    score_fn = (lambda v: float(np.mean(v)))
    result = af3_diffusion_step(x, t, score_fn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alf3df_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    score_fn = (lambda v: float(np.mean(v)))
    result = af3_diffusion_step(x, t, score_fn)
    assert isinstance(result, dict)
