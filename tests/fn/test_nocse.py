"""Tests for moirais.fn.nocse -- NOMINATE confidence intervals."""
import numpy as np
from moirais.fn.nocse import nominate_confidence_interval, nocse


def test_alias():
    assert nocse is nominate_confidence_interval


def test_smoke():
    se = np.array([0.1, 0.2, 0.15])
    r = nominate_confidence_interval(se)
    assert r.name == "nominate_confidence_interval"
    assert r.extra["alpha"] == 0.05
    assert r.extra["n_params"] == 3


def test_custom_alpha():
    r = nominate_confidence_interval([0.1], alpha=0.01)
    assert r.extra["alpha"] == 0.01
    assert r.extra["z_critical"] > 2.3
