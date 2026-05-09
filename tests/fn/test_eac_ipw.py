"""Tests for moirais.fn.eac_ipw — eBAC selection-adjusted IPW analysis."""

import numpy as np
import pandas as pd
import pytest

from moirais.fn.eac_ipw import run_ebac_selection_ipw_analysis


@pytest.fixture()
def synth_ebac_data():
    """Synthetic eBAC-like data with selection mechanism."""
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    treatment = rng.binomial(1, 0.35, n)
    alcohol = np.ones(n, dtype=int)  # all eligible

    # eBAC: some missing (selection), continuous outcome
    ebac_vals = np.abs(0.02 + 0.01 * treatment + 0.005 * x1 + rng.standard_normal(n) * 0.02)
    # Introduce missingness (selection)
    obs_prob = 1 / (1 + np.exp(-(0.5 + 0.3 * x1 - 0.2 * treatment)))
    observed = rng.binomial(1, obs_prob).astype(bool)
    ebac_vals_with_na = ebac_vals.copy()
    ebac_vals_with_na[~observed] = np.nan

    # Binary outcome: over legal limit (0.08)
    ebac_legal = (ebac_vals > 0.04).astype(int)
    ebac_legal_with_na = ebac_legal.astype(float)
    ebac_legal_with_na[~observed] = np.nan

    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "cannabis_any_use": treatment,
        "alcohol_past12m": alcohol,
        "ebac_tot": ebac_vals_with_na,
        "ebac_legal": ebac_legal_with_na,
        "weight": rng.uniform(0.5, 2.0, n),
    })


def test_returns_dict(synth_ebac_data):
    result = run_ebac_selection_ipw_analysis(
        synth_ebac_data,
        covariates=["x1", "x2"],
    )
    assert isinstance(result, dict)


def test_has_expected_keys(synth_ebac_data):
    result = run_ebac_selection_ipw_analysis(
        synth_ebac_data,
        covariates=["x1", "x2"],
    )
    expected_keys = {
        "analysis_frame",
        "ebac_ipw_weight_diagnostics",
        "ebac_ipw_logistic_or",
        "ebac_ipw_linear_coefficients",
        "ebac_ipw_cannabis_comparison",
        "ebac_final_ipw_diagnostics",
    }
    assert expected_keys.issubset(result.keys())


def test_diagnostics_is_dataframe(synth_ebac_data):
    result = run_ebac_selection_ipw_analysis(
        synth_ebac_data,
        covariates=["x1", "x2"],
    )
    diag = result["ebac_final_ipw_diagnostics"]
    assert isinstance(diag, pd.DataFrame)
    assert len(diag) > 0


def test_analysis_frame_only_observed(synth_ebac_data):
    result = run_ebac_selection_ipw_analysis(
        synth_ebac_data,
        covariates=["x1", "x2"],
    )
    frame = result["analysis_frame"]
    # All rows in analysis_frame should have non-missing ebac_tot
    assert frame["ebac_tot"].notna().all()


def test_logistic_or_has_treatment_term(synth_ebac_data):
    result = run_ebac_selection_ipw_analysis(
        synth_ebac_data,
        covariates=["x1", "x2"],
    )
    or_df = result["ebac_ipw_logistic_or"]
    assert "cannabis_any_use" in or_df["term"].values
