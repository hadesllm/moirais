"""Tests for morie.fn.pstcp -- compare posterior parameters."""
import numpy as np
from morie.fn.pstcp import posterior_compare_params, pstcp


def test_alias():
    assert pstcp is posterior_compare_params


def test_smoke():
    chain = np.random.default_rng(42).standard_normal((200, 4))
    r = posterior_compare_params(chain, param_indices=[0, 2])
    assert r.name == "posterior_compare_params"
    assert r.extra["n_params"] == 2
    assert len(r.extra["comparisons"]) == 2
    assert "mean" in r.extra["comparisons"][0]
    assert "ci_lo" in r.extra["comparisons"][0]
