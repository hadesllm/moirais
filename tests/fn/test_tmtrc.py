"""Tests for moirais.fn.tmtrc -- trace over time."""
import numpy as np
from moirais.fn.tmtrc import trace_over_time, tmtrc


def test_alias():
    assert tmtrc is trace_over_time


def test_smoke():
    rng = np.random.default_rng(42)
    sessions = [rng.standard_normal((10, 2)) for _ in range(3)]
    r = trace_over_time(sessions, ["S1", "S2", "S3"])
    assert r.name == "trace_over_time"
    assert r.extra["n_sessions"] == 3
    assert r.extra["n_legislators"] == 10
    assert len(r.extra["mean_shifts"]) == 2
