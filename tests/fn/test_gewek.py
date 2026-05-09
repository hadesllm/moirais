"""Tests for moirais.fn.gewek -- Geweke diagnostic."""

import numpy as np
from moirais.fn.gewek import geweke_diagnostic


def test_returns_dict():
    rng = np.random.default_rng(42)
    result = geweke_diagnostic(rng.normal(0, 1, 500))
    assert isinstance(result, dict)
    assert "z_score" in result
    assert "p_value" in result


def test_converged_chain():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 10000)
    result = geweke_diagnostic(samples)
    assert result["p_value"] > 0.01


def test_nonconverged_chain():
    samples = np.concatenate([np.zeros(500), np.ones(500) * 100])
    result = geweke_diagnostic(samples)
    assert result["converged"] is False


def test_p_value_in_range():
    rng = np.random.default_rng(42)
    result = geweke_diagnostic(rng.normal(0, 1, 500))
    assert 0 <= result["p_value"] <= 1


def test_too_short():
    try:
        geweke_diagnostic([1, 2, 3])
        assert False
    except ValueError:
        pass


def test_fractions_too_large():
    try:
        geweke_diagnostic(np.zeros(100), first_frac=0.6, last_frac=0.6)
        assert False
    except ValueError:
        pass
