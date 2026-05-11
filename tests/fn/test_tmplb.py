"""Tests for morie.fn.tmplb — template library matching."""
import numpy as np
import pytest

from morie.fn.tmplb import template_library, tmplb


def test_exact_match():
    rng = np.random.default_rng(42)
    template = rng.standard_normal(20)
    signal = np.concatenate([rng.standard_normal(30), template, rng.standard_normal(30)])
    result = template_library([template], signal, method="correlation")
    assert result.value > 0.9
    assert result.extra["best_index"] == 0


def test_euclidean_method():
    rng = np.random.default_rng(42)
    template = np.ones(10)
    signal = np.concatenate([np.zeros(10), np.ones(10), np.zeros(10)])
    result = template_library([template], signal, method="euclidean")
    assert result.value < 1e-10


def test_multiple_templates():
    rng = np.random.default_rng(42)
    t1 = np.ones(10)
    t2 = np.sin(np.linspace(0, 2 * np.pi, 10))
    signal = np.concatenate([np.zeros(5), np.ones(10), np.zeros(5)])
    result = template_library([t1, t2], signal, method="correlation")
    assert result.extra["n_templates"] == 2


def test_alias():
    assert tmplb is template_library
