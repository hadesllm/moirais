"""Tests for morie.fn.hausr — Hausman test."""

import numpy as np
import pytest

from morie.fn.hausr import hausman_test


def test_hausman_correlated_effects():
    rng = np.random.default_rng(42)
    N, T = 50, 10
    y, X, ent = [], [], []
    for i in range(N):
        alpha_i = rng.standard_normal()
        for t in range(T):
            x = rng.standard_normal() + alpha_i * 0.5
            y.append(alpha_i + 2.0 * x + rng.standard_normal() * 0.5)
            X.append([x])
            ent.append(i)
    res = hausman_test(np.array(y), np.array(X), np.array(ent))
    assert res.value >= 0
    assert res.extra["df"] == 1


def test_hausman_no_correlation():
    rng = np.random.default_rng(42)
    N, T = 50, 10
    y, X, ent = [], [], []
    for i in range(N):
        alpha_i = rng.standard_normal() * 0.1
        for t in range(T):
            x = rng.standard_normal()
            y.append(alpha_i + 2.0 * x + rng.standard_normal() * 0.5)
            X.append([x])
            ent.append(i)
    res = hausman_test(np.array(y), np.array(X), np.array(ent))
    assert res.value >= 0
    assert "interpretation" in res.extra


def test_hausman_has_both_betas():
    rng = np.random.default_rng(7)
    N, T = 20, 5
    y, X, ent = [], [], []
    for i in range(N):
        for t in range(T):
            x = rng.standard_normal()
            y.append(1.0 + x + rng.standard_normal() * 0.5)
            X.append([x])
            ent.append(i)
    res = hausman_test(np.array(y), np.array(X), np.array(ent))
    assert len(res.extra["beta_fe"]) == 1
    assert len(res.extra["beta_re"]) == 1
