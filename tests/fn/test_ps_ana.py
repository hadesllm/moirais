"""Tests for moirais.fn.ps_ana — propensity score IPW analysis pipeline."""

import numpy as np
import pandas as pd
import pytest

from moirais.fn.ps_ana import run_propensity_ipw_analysis


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.3 * x1 - 0.2 * x2)))
    t = rng.binomial(1, prob)
    y = 0.5 * t + 0.3 * x1 + rng.standard_normal(n) * 0.5
    y_bin = (y > 0.3).astype(int)
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "treatment": t,
        "outcome": y_bin,
        "weight": rng.uniform(0.5, 2.0, n),
    })


def test_returns_dict_with_expected_keys(synth_data):
    result = run_propensity_ipw_analysis(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1", "x2"],
        survey_weight_col="weight",
    )
    assert isinstance(result, dict)
    assert "analysis_frame" in result
    assert "ipw_results" in result
    assert "diagnostics" in result


def test_ipw_results_has_ate(synth_data):
    result = run_propensity_ipw_analysis(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1", "x2"],
        survey_weight_col="weight",
    )
    ipw_df = result["ipw_results"]
    assert "estimate" in ipw_df.columns
    assert np.isfinite(ipw_df["estimate"].iloc[0])


def test_diagnostics_has_ps_metrics(synth_data):
    result = run_propensity_ipw_analysis(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1", "x2"],
        survey_weight_col="weight",
    )
    diag = result["diagnostics"]
    metrics = diag["metric"].tolist()
    assert "ps_mean" in metrics
    assert "ess_ipw_trimmed" in metrics


def test_analysis_frame_has_ps_and_ipw_columns(synth_data):
    result = run_propensity_ipw_analysis(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1", "x2"],
        survey_weight_col="weight",
    )
    frame = result["analysis_frame"]
    assert "ps" in frame.columns
    assert "ipw" in frame.columns
    assert "ipw_trimmed" in frame.columns
