"""Tests for moirais.fn.irtva -- per-legislator IRT variance."""
import numpy as np
from moirais.fn.irtva import irt_variance_legislator, irtva


def test_alias():
    assert irtva is irt_variance_legislator


def test_smoke():
    chain = np.random.default_rng(42).standard_normal((200, 5))
    r = irt_variance_legislator(chain)
    assert r.name == "irt_variance_legislator"
    assert r.extra["n_legislators"] == 5
    assert r.extra["n_samples"] == 200
    assert all(v > 0 for v in r.extra["variances"])
