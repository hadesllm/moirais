"""Tests for moirais.fn.bplag — Breusch-Pagan LM test."""

import numpy as np
import pytest

from moirais.fn.bplag import breusch_pagan_lm


def test_bp_detects_random_effects():
    rng = np.random.default_rng(42)
    N, T = 50, 10
    y, X, ent = [], [], []
    for i in range(N):
        alpha_i = rng.standard_normal() * 3.0
        for t in range(T):
            x = rng.standard_normal()
            y.append(1.0 + 2.0 * x + alpha_i + rng.standard_normal() * 0.5)
            X.append([x])
            ent.append(i)
    res = breusch_pagan_lm(np.array(y), np.array(X), np.array(ent))
    assert res.extra["reject_H0"] is True


def test_bp_no_effects():
    rng = np.random.default_rng(42)
    N, T = 50, 10
    y, X, ent = [], [], []
    for i in range(N):
        for t in range(T):
            x = rng.standard_normal()
            y.append(1.0 + 2.0 * x + rng.standard_normal() * 0.5)
            X.append([x])
            ent.append(i)
    res = breusch_pagan_lm(np.array(y), np.array(X), np.array(ent))
    assert res.value >= 0
    assert res.extra["n_entities"] == N


def test_bp_df_is_1():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(100)
    X = rng.standard_normal((100, 1))
    ent = np.repeat(np.arange(20), 5)
    res = breusch_pagan_lm(y, X, ent)
    assert res.extra["df"] == 1
