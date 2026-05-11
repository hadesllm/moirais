"""Tests for morie.fn.tgtpr — targeting parameter."""

import numpy as np
import pytest

from morie.fn.tgtpr import tgtpr


def test_logit_link():
    rng = np.random.default_rng(42)
    n = 200
    Q_init = np.clip(rng.uniform(0.2, 0.8, n), 0.01, 0.99)
    H = rng.standard_normal(n)
    Y = rng.binomial(1, Q_init).astype(float)
    result = tgtpr(Y, Q_init, H, link="logit")
    assert "epsilon" in result
    assert "Q_star" in result


def test_identity_link():
    rng = np.random.default_rng(7)
    n = 200
    Q_init = rng.standard_normal(n) + 5
    H = rng.standard_normal(n)
    Y = Q_init + rng.standard_normal(n) * 0.1
    result = tgtpr(Y, Q_init, H, link="identity")
    assert result["converged"] is True


def test_qstar_shape():
    n = 50
    result = tgtpr(np.ones(n) * 0.5, np.ones(n) * 0.5, np.ones(n), link="identity")
    assert len(result["Q_star"]) == n
