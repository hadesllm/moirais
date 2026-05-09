"""Tests for hrvfd — HRV frequency-domain metrics."""
import numpy as np
from moirais.fn.hrvfd import hrv_freq_domain
from moirais.fn._containers import DescriptiveResult


def test_hrvfd_basic(rng):
    rr = 800 + rng.standard_normal(200) * 50
    result = hrv_freq_domain(rr)
    assert isinstance(result, DescriptiveResult)
    assert "vlf" in result.extra
    assert "lf" in result.extra
    assert "hf" in result.extra


def test_hrvfd_power_positive(rng):
    rr = 800 + rng.standard_normal(200) * 50
    result = hrv_freq_domain(rr)
    assert result.extra["vlf"] >= 0
    assert result.extra["lf"] >= 0
    assert result.extra["hf"] >= 0
