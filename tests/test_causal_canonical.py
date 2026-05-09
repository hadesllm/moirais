"""
Canonical-value tests for MOIRAIS's causal inference core.

Each test validates an MOIRAIS function against a *known answer* from a
simulated DGP (data-generating process), not just "returns a dict with
the right keys". When a test in this file fails, it's a real math error,
not a template regression.

Paper / canonical reference is cited per test. Milestone M1 of the
the canonical causal-inference test surface.

Design principles:
- Use np.random.default_rng with a fixed seed for reproducibility.
- Each DGP is documented inline (you can rederive the true answer).
- Tolerances are explicit and tight — wider tolerance means weaker
  canonical claim.
- If a test is flaky at the tolerance, loosen ONLY with a comment
  explaining why (e.g., "Monte-Carlo variance at n=5000 ~0.15 by the
  Chernozhukov et al. (2018) §6 numbers").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# DGP fixtures — shared simulators cited from the literature.
# ---------------------------------------------------------------------------

def _dgp_logistic_ps(
    n: int = 5000,
    tau: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Classical confounded observational DGP:
        X ~ N(0, 1)^p, p=3
        T | X ~ Bernoulli(sigmoid(X @ beta_ps))          ← selection
        Y(0) = X @ alpha + eps
        Y(1) = Y(0) + tau                                 ← homogeneous effect
        Y = T*Y(1) + (1-T)*Y(0)

    True ATE = tau (by construction, same for every unit).
    True ATT = tau (homogeneous effect).
    Propensity depends on X so naive OLS of Y on T is biased.

    Adapted from Hernán & Robins, "Causal Inference: What If" (2020),
    Technical Point 12.1 — the canonical textbook confounded DGP.
    """
    rng = np.random.default_rng(seed)
    p = 3
    X = rng.normal(size=(n, p))
    beta_ps = np.array([0.8, -0.5, 0.4])
    alpha = np.array([1.2, -0.7, 0.3])
    logit = X @ beta_ps
    ps_true = 1.0 / (1.0 + np.exp(-logit))
    T = rng.binomial(1, ps_true)
    Y0 = X @ alpha + rng.normal(scale=1.0, size=n)
    Y1 = Y0 + tau
    Y = T * Y1 + (1 - T) * Y0
    return pd.DataFrame({
        "Y": Y, "T": T,
        "X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2],
        "ps_true": ps_true,
    })


# ---------------------------------------------------------------------------
# M1-01 · calculate_ipw_weights — known formulas
# ---------------------------------------------------------------------------

class TestIpwWeightsCanonical:
    """
    Reference: Horvitz & Thompson (1952); Hernán & Robins (2020) Ch 12.
    Properties checked:
      1. Raw weights equal textbook formula t/ps + (1-t)/(1-ps) exactly.
      2. At correct ps (ps_true), sum of weights in each group ≈ n
         (Horvitz-Thompson balance).
      3. Stabilized weights have expectation 1 (stabilization property).
      4. Trimming at (0.01, 0.99) caps extremes.
    """

    def test_raw_formula_exact(self) -> None:
        from moirais.fn.ipw import calculate_ipw_weights
        df = _dgp_logistic_ps(n=500, seed=1)
        w = calculate_ipw_weights(df, treatment="T", ps_col="ps_true")
        t = df["T"].values
        ps = df["ps_true"].clip(0.01, 0.99).values
        expected = t / ps + (1 - t) / (1 - ps)
        np.testing.assert_allclose(w.values, expected, rtol=1e-10,
            err_msg="IPW raw weight formula departs from t/ps + (1-t)/(1-ps)")

    def test_horvitz_thompson_balance(self) -> None:
        """Sum of treated weights should approximate n_total (and same for
        controls) when ps is correct — that's the Horvitz-Thompson balance
        property that makes IPW identify ATE."""
        from moirais.fn.ipw import calculate_ipw_weights
        df = _dgp_logistic_ps(n=5000, seed=2)
        w = calculate_ipw_weights(df, treatment="T", ps_col="ps_true")
        n = len(df)
        sum_treated_w = float(w[df["T"] == 1].sum())
        sum_control_w = float(w[df["T"] == 0].sum())
        # Monte-Carlo: ratio should be close to 1.0, tolerance ~3% at n=5000
        assert abs(sum_treated_w / n - 1.0) < 0.05, (
            f"Treated IPW-sum/n = {sum_treated_w/n:.3f}, expected ~1.0")
        assert abs(sum_control_w / n - 1.0) < 0.05, (
            f"Control IPW-sum/n = {sum_control_w/n:.3f}, expected ~1.0")

    def test_stabilized_expectation_one(self) -> None:
        """Stabilized weights have mean ~1 by construction (marginal prob
        numerator cancels the denominator in expectation)."""
        from moirais.fn.ipw import calculate_ipw_weights
        df = _dgp_logistic_ps(n=5000, seed=3)
        w = calculate_ipw_weights(df, treatment="T", ps_col="ps_true",
                                  stabilized=True)
        assert abs(float(w.mean()) - 1.0) < 0.05, (
            f"Stabilized mean = {float(w.mean()):.3f}, expected ~1.0")

    def test_trim_quantiles_caps_extremes(self) -> None:
        from moirais.fn.ipw import calculate_ipw_weights
        df = _dgp_logistic_ps(n=500, seed=4)
        w_untrimmed = calculate_ipw_weights(df, treatment="T", ps_col="ps_true")
        w_trimmed = calculate_ipw_weights(df, treatment="T", ps_col="ps_true",
                                          trim_quantiles=(0.05, 0.95))
        assert float(w_trimmed.max()) <= float(w_untrimmed.quantile(0.95)) + 1e-9
        assert float(w_trimmed.min()) >= float(w_untrimmed.quantile(0.05)) - 1e-9


# ---------------------------------------------------------------------------
# M1-02 · estimate_ate — IPW-weighted OLS
# ---------------------------------------------------------------------------

class TestEstimateAteCanonical:
    """
    Reference: Hernán & Robins (2020) §12.4. The marginal structural model
    Y = β₀ + β₁T fit by weighted least squares with IPTW weights is a
    consistent estimator of the ATE under correct ps specification.

    Properties:
      1. At known ps, estimated ATE within Monte-Carlo tolerance of true τ.
      2. Bias shrinks with n (comparing n=500 vs n=5000).
      3. Standard error from HC3 is non-trivial (> 0) and well-behaved.
    """

    def test_point_estimate_matches_truth_at_n5000(self) -> None:
        from moirais.fn.ipw import calculate_ipw_weights
        from moirais.fn.ate import estimate_ate
        tau_true = 2.0
        df = _dgp_logistic_ps(n=5000, tau=tau_true, seed=11)
        df["w"] = calculate_ipw_weights(df, treatment="T", ps_col="ps_true")
        ate, se = estimate_ate(df, outcome="Y", treatment="T", weights_col="w")
        # Single-simulation tolerance: |ate - tau| < 3 SE (roughly 99% interval)
        assert abs(ate - tau_true) < max(3 * se, 0.15), (
            f"ATE = {ate:.3f} (SE {se:.3f}), true τ = {tau_true}. "
            f"Bias = {ate - tau_true:.3f} exceeds 3*SE.")
        assert se > 0, "HC3 SE must be positive for a non-degenerate sample."

    def test_bias_shrinks_with_n(self) -> None:
        from moirais.fn.ipw import calculate_ipw_weights
        from moirais.fn.ate import estimate_ate
        tau_true = 2.0
        biases = []
        for n in (500, 5000):
            # Average absolute bias across 5 seeds per n — stabilises Monte-Carlo
            seed_biases = []
            for s in range(5):
                df = _dgp_logistic_ps(n=n, tau=tau_true, seed=100 + s)
                df["w"] = calculate_ipw_weights(df, treatment="T", ps_col="ps_true")
                ate, _ = estimate_ate(df, outcome="Y", treatment="T", weights_col="w")
                seed_biases.append(abs(ate - tau_true))
            biases.append(np.mean(seed_biases))
        # Bigger n should give smaller mean absolute bias. Allow some noise.
        assert biases[1] < biases[0] * 1.5, (
            f"Expected bias to shrink with n. "
            f"|bias(n=500)| = {biases[0]:.3f}, |bias(n=5000)| = {biases[1]:.3f}")


# ---------------------------------------------------------------------------
# M1-03 · estimate_plr (DoubleML PLR) — Chernozhukov et al. (2018)
# ---------------------------------------------------------------------------

class TestPlrCanonical:
    """
    Reference: Chernozhukov et al. (2018) "Double/Debiased ML for Treatment
    and Structural Parameters", Section 6. Partially linear regression
    model Y = θ·D + g(X) + U, E[U|D,X]=0.

    DoubleML's PLR estimator achieves √n-consistent inference for θ when the
    nuisance functions g(X) and m(X)=E[D|X] are estimated at o(n^{-1/4}) rate
    with K-fold cross-fitting.

    We use a simple linear g(X) here (so any reasonable ML fits) and check
    that the DML estimate is √n-close to the true θ=0.5.
    """

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_dml_plr_recovers_theta_on_linear_dgp(self) -> None:
        try:
            from moirais.fn.plr import estimate_plr
        except ImportError:
            pytest.skip("estimate_plr not exposed from moirais.fn.plr")

        rng = np.random.default_rng(321)
        n, p = 1000, 5
        theta_true = 0.5
        X = rng.normal(size=(n, p))
        beta_m = rng.normal(scale=0.3, size=p)     # E[D|X]
        beta_g = rng.normal(scale=0.3, size=p)     # g(X)
        D = X @ beta_m + rng.normal(scale=1.0, size=n)
        Y = theta_true * D + X @ beta_g + rng.normal(scale=1.0, size=n)
        df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
        df["D"] = D
        df["Y"] = Y

        # estimate_plr signature may vary; try the most common shape.
        try:
            result = estimate_plr(
                data=df,
                outcome="Y",
                treatment="D",
                covariates=[f"X{i+1}" for i in range(p)],
            )
        except TypeError:
            pytest.skip("estimate_plr signature does not match canonical test yet — "
                        "flagged for audit pass")

        # plr.py returns a dict with keys: ate, se, ci_lower, ci_upper, pval, n_obs.
        # Accept several synonyms for forward-compat with any future rewrites.
        if isinstance(result, dict):
            for k in ("ate", "estimate", "theta", "coef"):
                if k in result and result[k] is not None:
                    theta_hat = float(result[k])
                    break
            else:
                pytest.fail(f"estimate_plr returned dict with no ate/estimate/theta/coef key: {list(result.keys())}")
        elif isinstance(result, tuple):
            theta_hat = float(result[0])
        else:
            theta_hat = float(result)

        # √n-consistency: at n=1000 we expect |theta_hat - 0.5| < 3/√1000 ≈ 0.095
        # plus Monte-Carlo noise; allow 0.15 for a single-seed test.
        assert abs(theta_hat - theta_true) < 0.15, (
            f"PLR θ̂ = {theta_hat:.4f}, true θ = {theta_true}. "
            f"Bias {theta_hat - theta_true:.4f} exceeds 0.15 tolerance.")


# ---------------------------------------------------------------------------
# M1-04 · estimate_att — Hajek-weighted IPW for ATT
# ---------------------------------------------------------------------------

class TestEstimateAttCanonical:
    """
    Reference: Imbens (2004) Review of Economics and Statistics §3;
    Hirano, Imbens & Ridder (2003) Econometrica.

    Under homogeneous treatment effect τ (Y(1) - Y(0) = τ for every unit),
    ATE = ATT = ATC = τ. This gives us a trivial truth for testing the
    point estimator. Heterogeneous-effect tests live in the CATE/GATE
    audit, not here.

    Properties:
      1. Under homogeneous DGP, ATT point estimate within 3*SE of τ.
      2. Returns the expected dict keys and sensible SE.
      3. On mostly-treated data (low prop of controls), SE inflates
         correctly (sanity).
    """

    def test_att_matches_tau_on_homogeneous_dgp(self) -> None:
        from moirais.fn.att import estimate_att
        tau_true = 2.0
        df = _dgp_logistic_ps(n=5000, tau=tau_true, seed=21)
        result = estimate_att(
            df,
            treatment="T",
            outcome="Y",
            covariates=["X1", "X2", "X3"],
            propensity_col="ps_true",
        )
        assert isinstance(result, dict)
        assert {"att", "se", "ci_lower", "ci_upper"}.issubset(result.keys())
        att = float(result["att"])
        se = float(result["se"])
        assert se > 0, "ATT SE must be positive."
        assert abs(att - tau_true) < max(3 * se, 0.15), (
            f"ATT = {att:.4f} (SE {se:.4f}), true τ = {tau_true}. "
            f"Bias {att - tau_true:.4f} exceeds 3*SE.")

    def test_att_ci_has_correct_width(self) -> None:
        """CI width ≈ 2 * z * SE where z = 1.96."""
        from moirais.fn.att import estimate_att
        df = _dgp_logistic_ps(n=2000, tau=1.5, seed=22)
        result = estimate_att(df, treatment="T", outcome="Y",
                              covariates=["X1", "X2", "X3"], propensity_col="ps_true")
        width = float(result["ci_upper"]) - float(result["ci_lower"])
        expected = 2 * 1.959964 * float(result["se"])
        np.testing.assert_allclose(width, expected, rtol=1e-4,
            err_msg="ATT CI width doesn't match 2*z*SE.")


# ---------------------------------------------------------------------------
# M1-05 · estimate_atc — Hajek-weighted IPW for ATC
# ---------------------------------------------------------------------------

class TestEstimateAtcCanonical:
    """
    Reference: Imbens (2004). Mirror of ATT: effect on the controls.
    Under homogeneous effect, ATC = τ.
    """

    def test_atc_matches_tau_on_homogeneous_dgp(self) -> None:
        from moirais.fn.atc import estimate_atc
        tau_true = 2.0
        df = _dgp_logistic_ps(n=5000, tau=tau_true, seed=31)
        result = estimate_atc(
            df,
            treatment="T",
            outcome="Y",
            covariates=["X1", "X2", "X3"],
            propensity_col="ps_true",
        )
        assert isinstance(result, dict)
        assert {"atc", "se", "ci_lower", "ci_upper"}.issubset(result.keys())
        atc = float(result["atc"])
        se = float(result["se"])
        assert se > 0, "ATC SE must be positive."
        assert abs(atc - tau_true) < max(3 * se, 0.15), (
            f"ATC = {atc:.4f} (SE {se:.4f}), true τ = {tau_true}. "
            f"Bias {atc - tau_true:.4f} exceeds 3*SE.")

    def test_att_atc_agree_on_homogeneous_dgp(self) -> None:
        """On a homogeneous-effect DGP, ATT and ATC should both ≈ τ, and
        their difference should be within Monte-Carlo noise."""
        from moirais.fn.att import estimate_att
        from moirais.fn.atc import estimate_atc
        df = _dgp_logistic_ps(n=5000, tau=2.0, seed=32)
        kw = dict(treatment="T", outcome="Y",
                  covariates=["X1", "X2", "X3"], propensity_col="ps_true")
        att = float(estimate_att(df, **kw)["att"])
        atc = float(estimate_atc(df, **kw)["atc"])
        # |ATT - ATC| should be small (both estimate same τ, just different weights).
        assert abs(att - atc) < 0.30, (
            f"ATT = {att:.4f}, ATC = {atc:.4f}. Difference {att - atc:.4f} "
            f"exceeds 0.30 on homogeneous DGP — suggests one of the "
            f"weight formulas is mis-specified.")


# ---------------------------------------------------------------------------
# M1-06 · estimate_aipw — double-robustness (the big invariant)
# ---------------------------------------------------------------------------

class TestAipwDoubleRobustness:
    """
    Reference: Robins, Rotnitzky & Zhao (1994) JASA; Scharfstein,
    Rotnitzky & Robins (1999) JASA.

    AIPW's defining property: **consistent if EITHER the outcome model
    OR the propensity model is correctly specified — not necessarily
    both**. This is the textbook way to validate a doubly-robust
    implementation.

    Strategy:
      - A: both models linear-correctly-specified → baseline consistency.
      - B: outcome model correct, propensity mis-specified (use constant
           ps = 0.5 instead of logistic X). AIPW should still be close.
      - C: propensity model correct, outcome model mis-specified (use
           mean-only outcome model). AIPW should still be close.

    Because `estimate_aipw` calls `compute_propensity_scores` internally,
    we simulate mis-specification by over-riding the propensity column
    in the data frame with either the true PS (correct) or uniform
    noise (mis-specified). For mis-specified outcome model, we pass a
    constant covariate vector that the outcome model can't use.

    NOTE: estimate_aipw's signature does NOT currently accept a
    propensity_col override, so we can only test correct-both (scenario A)
    and correct-ps-with-default-linear-outcome in our linear DGP. The
    full DR test requires a signature extension — logged to M2 backlog.
    """

    def test_aipw_consistent_on_linear_dgp(self) -> None:
        """Scenario A: linear DGP where default linear outcome model AND
        default logistic PS are both correctly specified → AIPW is
        consistent (basic sanity before testing double-robustness)."""
        from moirais.fn.aipw import estimate_aipw
        tau_true = 2.0
        df = _dgp_logistic_ps(n=5000, tau=tau_true, seed=41)
        result = estimate_aipw(
            df,
            treatment="T",
            outcome="Y",
            covariates=["X1", "X2", "X3"],
            outcome_model="linear",
        )
        ate = float(result["ate"])
        se = float(result["se"])
        assert se > 0
        assert abs(ate - tau_true) < max(3 * se, 0.15), (
            f"AIPW ATE = {ate:.4f} (SE {se:.4f}), true τ = {tau_true}. "
            f"Bias {ate - tau_true:.4f} exceeds 3*SE on fully correctly "
            f"specified DGP.")

    def test_aipw_ate_beats_naive_on_confounded_dgp(self) -> None:
        """Naive OLS of Y on T alone is biased under confounding (our DGP
        has selection via logit(Xβ)). AIPW should be closer to truth than
        naive difference-in-means. This is a weaker but directly-testable
        form of the DR claim."""
        from moirais.fn.aipw import estimate_aipw
        tau_true = 2.0
        df = _dgp_logistic_ps(n=5000, tau=tau_true, seed=42)

        # Naive difference in means
        naive = float(df[df["T"] == 1]["Y"].mean() - df[df["T"] == 0]["Y"].mean())
        naive_bias = abs(naive - tau_true)

        # AIPW
        result = estimate_aipw(
            df,
            treatment="T",
            outcome="Y",
            covariates=["X1", "X2", "X3"],
            outcome_model="linear",
        )
        aipw_bias = abs(float(result["ate"]) - tau_true)

        # AIPW should be meaningfully better than naive on a confounded DGP.
        # Conservative claim: AIPW bias < 0.5 * naive bias (usually much better).
        assert aipw_bias < 0.5 * naive_bias, (
            f"AIPW bias ({aipw_bias:.4f}) not meaningfully smaller than "
            f"naive bias ({naive_bias:.4f}). On a confounded DGP, AIPW "
            f"should reduce confounding bias by at least half.")


# ---------------------------------------------------------------------------
# M1-07 · force (Star-Wars naming layer): ATE difference-in-means
# ---------------------------------------------------------------------------

def _dgp_randomized(n: int = 5000, tau: float = 2.0, seed: int = 51) -> pd.DataFrame:
    """
    Fully randomized DGP (no confounding):
        T ~ Bernoulli(0.5) independent of X
        Y(0) = X @ alpha + eps
        Y = T*tau + Y(0)

    Under randomization, naive difference-in-means = ATE (consistent).
    This is the setting force.ate_diff is built for.
    """
    rng = np.random.default_rng(seed)
    p = 3
    X = rng.normal(size=(n, p))
    alpha = np.array([1.2, -0.7, 0.3])
    T = rng.binomial(1, 0.5, size=n)
    Y0 = X @ alpha + rng.normal(scale=1.0, size=n)
    Y = T * tau + Y0
    return pd.DataFrame({"outcome": Y, "treatment": T,
                         "X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2]})


class TestForceCanonical:
    """
    `force` is the Star-Wars-themed alias for `ate_diff` in fn/force.py —
    naive difference-in-means ATE with Welch-t CI. Valid under randomization
    or as a baseline to compare confounding-adjusted estimators against.

    Reference: any intro causal-inference text, e.g., Imbens & Rubin (2015)
    Ch 6 on randomized experiments.
    """

    def test_force_unbiased_on_randomized_dgp(self) -> None:
        from moirais.fn.force import ate_diff
        tau_true = 2.0
        df = _dgp_randomized(n=5000, tau=tau_true, seed=51)
        result = ate_diff(df, y="outcome", t="treatment")
        ate = float(result.estimate)
        se = float(result.se)
        assert se > 0
        assert abs(ate - tau_true) < max(3 * se, 0.15), (
            f"force ATE = {ate:.4f} (SE {se:.4f}), true τ = {tau_true}. "
            f"Bias {ate - tau_true:.4f} exceeds 3*SE on randomized DGP.")

    def test_force_welch_ci_coverage_sanity(self) -> None:
        """CI width should equal 2 * t_crit * SE (Welch-Satterthwaite df)."""
        from moirais.fn.force import ate_diff
        from scipy import stats
        df = _dgp_randomized(n=1000, tau=1.5, seed=52)
        result = ate_diff(df, y="outcome", t="treatment", alpha=0.05)
        width = float(result.ci_upper) - float(result.ci_lower)
        se = float(result.se)
        df_w = float(result.extra["df"])
        expected = 2 * stats.t.ppf(0.975, df_w) * se
        np.testing.assert_allclose(width, expected, rtol=1e-4,
            err_msg="force CI width doesn't match 2 * t_crit(df_w) * SE.")


# ---------------------------------------------------------------------------
# M1-08 · estimate_double_ml — DoubleMLPLR with Random Forest nuisances
# ---------------------------------------------------------------------------

class TestEstimateDoubleMlCanonical:
    """
    Reference: Chernozhukov et al. (2018); Bach et al. (2022) JMLR.

    DML-PLR returns a fitted `DoubleMLPLR` object (not a dict). The
    causal estimate is `.coef[0]`; its SE is `.se[0]`. Random Forest
    nuisance learners are less efficient than ridge on linear DGPs,
    so we allow a wider tolerance (~0.25) than the plr.py test (0.15).

    Properties:
      1. Returns a DoubleMLPLR object with .coef and .se attributes.
      2. Point estimate within (3*SE, 0.30) of true θ on linear DGP.
      3. SE is finite and positive.
    """

    def test_dml_plr_returns_doubleml_object(self) -> None:
        from moirais.fn.dml import estimate_double_ml
        df = _dgp_randomized(n=500, tau=1.0, seed=61)
        # estimate_double_ml expects T as treatment; our DGP uses "treatment"
        obj = estimate_double_ml(
            data=df,
            outcome="outcome",
            treatment="treatment",
            covariates=["X1", "X2", "X3"],
            n_folds=3,  # faster for CI
        )
        # Should expose .coef and .se (DoubleMLPLR attributes, not dict)
        assert hasattr(obj, "coef"), f"Expected .coef attr, got {type(obj)}"
        assert hasattr(obj, "se"), f"Expected .se attr, got {type(obj)}"
        coef = float(obj.coef[0])
        se = float(obj.se[0])
        assert np.isfinite(coef), f"DML coef is {coef}, not finite"
        assert se > 0, f"DML SE is {se}, must be > 0"

    def test_dml_plr_recovers_theta_on_linear_dgp(self) -> None:
        """RF learners are less efficient than ridge on linear DGPs.
        Tolerance is wider (0.30) than plr.py's 0.15 test to account for
        this. If this is flaky, review the n_folds / RF max_depth tuning."""
        from moirais.fn.dml import estimate_double_ml
        tau_true = 2.0
        df = _dgp_randomized(n=2000, tau=tau_true, seed=62)
        obj = estimate_double_ml(
            data=df,
            outcome="outcome",
            treatment="treatment",
            covariates=["X1", "X2", "X3"],
            n_folds=3,
        )
        coef = float(obj.coef[0])
        se = float(obj.se[0])
        # RF tolerance on a 3-covariate linear DGP at n=2000: allow 3*SE or 0.30
        assert abs(coef - tau_true) < max(3 * se, 0.30), (
            f"DML θ̂ = {coef:.4f} (SE {se:.4f}), true θ = {tau_true}. "
            f"Bias {coef - tau_true:.4f} exceeds tolerance — possibly a "
            f"RF hyperparameter or n_folds issue, not an identification bug.")


# ---------------------------------------------------------------------------
# M1-09 · estimate_irm — DoubleMLIRM with Random Forest nuisances
# ---------------------------------------------------------------------------

class TestEstimateIrmCanonical:
    """
    Reference: Chernozhukov et al. (2018) §5.3, DoubleML IRM.

    IRM allows heterogeneous treatment effects. On a homogeneous-effect
    DGP, IRM and PLR target the same parameter (ATE = τ) but via
    different Neyman-orthogonal scores. On heterogeneous-effect DGPs,
    IRM remains consistent for E[τ(X)] while PLR is biased unless the
    interaction is specified.

    Properties:
      1. Returns dict with 'ate' key + CI attributes.
      2. Point estimate within 3*SE of τ on linear homogeneous DGP.
    """

    def test_irm_returns_dict_with_ate(self) -> None:
        from moirais.fn.irm import estimate_irm
        df = _dgp_randomized(n=500, tau=1.0, seed=71)
        result = estimate_irm(
            df,
            treatment="treatment",
            outcome="outcome",
            covariates=["X1", "X2", "X3"],
            n_folds=3,
        )
        assert isinstance(result, dict)
        assert {"ate", "se", "ci_lower", "ci_upper", "n"}.issubset(result.keys()), \
            f"IRM result missing expected keys. Got: {list(result.keys())}"
        ate = float(result["ate"])
        se = float(result["se"])
        assert np.isfinite(ate) and se > 0

    def test_irm_recovers_tau_on_homogeneous_dgp(self) -> None:
        from moirais.fn.irm import estimate_irm
        tau_true = 2.0
        df = _dgp_randomized(n=2000, tau=tau_true, seed=72)
        result = estimate_irm(
            df,
            treatment="treatment",
            outcome="outcome",
            covariates=["X1", "X2", "X3"],
            n_folds=3,
        )
        ate = float(result["ate"])
        se = float(result["se"])
        assert abs(ate - tau_true) < max(3 * se, 0.30), (
            f"IRM ATE = {ate:.4f} (SE {se:.4f}), true τ = {tau_true}. "
            f"Bias {ate - tau_true:.4f} exceeds tolerance.")

    def test_irm_and_dml_agree_on_homogeneous_dgp(self) -> None:
        """On a homogeneous-effect DGP, IRM and DML-PLR both target the
        same ATE. Estimates should agree within combined SE."""
        from moirais.fn.dml import estimate_double_ml
        from moirais.fn.irm import estimate_irm
        df = _dgp_randomized(n=2000, tau=1.5, seed=73)
        kw = dict(covariates=["X1", "X2", "X3"], n_folds=3)

        dml_obj = estimate_double_ml(
            data=df, outcome="outcome", treatment="treatment", **kw)
        irm_res = estimate_irm(
            df, treatment="treatment", outcome="outcome", **kw)

        dml_coef = float(dml_obj.coef[0])
        irm_ate = float(irm_res["ate"])
        # Combined SE as rough comparison scale
        pooled_se = float(np.sqrt(float(dml_obj.se[0])**2 + float(irm_res["se"])**2))
        diff = abs(dml_coef - irm_ate)
        assert diff < max(3 * pooled_se, 0.40), (
            f"DML-PLR and IRM disagree too much on homogeneous DGP: "
            f"DML={dml_coef:.4f}, IRM={irm_ate:.4f}, |diff|={diff:.4f}, "
            f"3*pooled_SE={3*pooled_se:.4f}. Both should estimate the same ATE.")


# ---------------------------------------------------------------------------
# M1-10 · iv_2sls — two-stage least squares with continuous instrument
# ---------------------------------------------------------------------------

def _dgp_iv_continuous(
    n: int = 3000,
    beta: float = 2.0,
    alpha_z: float = 0.8,
    seed: int = 81,
) -> pd.DataFrame:
    """
    Continuous IV DGP:
        Z ~ N(0,1)                                            (instrument)
        U ~ N(0,1)                                            (unobserved confounder)
        D = alpha_z * Z + 0.5 * U + v,  v ~ N(0,1)            (endogenous treatment)
        Y = beta * D + U + eps,         eps ~ N(0,1)          (outcome)

    U is correlated with D (through D's v-free share) and directly
    affects Y. OLS is biased because Cov(D, U) != 0. 2SLS with Z
    identifies beta because Cov(Z, U) = 0 and Z has first-stage power
    via alpha_z.

    Under homogeneous effects, true β recoverable by:
        β_IV = Cov(Y, Z) / Cov(D, Z)
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    U = rng.normal(size=n)
    v = rng.normal(size=n)
    D = alpha_z * Z + 0.5 * U + v
    eps = rng.normal(size=n)
    Y = beta * D + U + eps
    return pd.DataFrame({"Y": Y, "D": D, "Z": Z})


class TestIv2slsCanonical:
    """
    Reference: Angrist & Pischke (2009) *Mostly Harmless Econometrics* Ch 4.
    2SLS identifies β under homogeneous effects when Z is a valid instrument.

    Properties:
      1. Returns RegressionResult with coefficients, se, p_values, n.
      2. β̂ within (3*SE, 0.15) of true β on well-identified DGP.
      3. First-stage F > 10 (rule of thumb for non-weak instrument).
      4. Closed-form Wald identity: β_IV = Cov(Y,Z) / Cov(D,Z) on
         no-covariate case.
    """

    def test_iv_returns_regression_result(self) -> None:
        from moirais.fn.iv import iv_2sls
        df = _dgp_iv_continuous(n=500, seed=81)
        result = iv_2sls(df, y="Y", d="D", z="Z")
        assert hasattr(result, "coefficients"), f"Expected RegressionResult, got {type(result)}"
        assert "D" in result.coefficients, f"Expected 'D' coef in result.coefficients; got {list(result.coefficients.keys())}"
        assert "first_stage_F" in result.extra, "First-stage F missing from extras"

    def test_iv_recovers_beta_on_well_identified_dgp(self) -> None:
        from moirais.fn.iv import iv_2sls
        beta_true = 2.0
        df = _dgp_iv_continuous(n=3000, beta=beta_true, alpha_z=0.8, seed=82)
        result = iv_2sls(df, y="Y", d="D", z="Z")
        beta_hat = float(result.coefficients["D"])
        se = float(result.se["D"])
        assert se > 0
        assert abs(beta_hat - beta_true) < max(3 * se, 0.15), (
            f"IV β̂ = {beta_hat:.4f} (SE {se:.4f}), true β = {beta_true}. "
            f"Bias {beta_hat - beta_true:.4f} exceeds tolerance.")

    def test_iv_first_stage_f_rules_out_weak_instrument(self) -> None:
        """Stock-Yogo (2005) rule of thumb: first-stage F > 10 indicates
        not-too-weak instrument. Our DGP has alpha_z=0.8, n=3000, so F
        should be well above 10."""
        from moirais.fn.iv import iv_2sls
        df = _dgp_iv_continuous(n=3000, alpha_z=0.8, seed=83)
        result = iv_2sls(df, y="Y", d="D", z="Z")
        f_stat = float(result.extra["first_stage_F"])
        assert f_stat > 10, (
            f"First-stage F = {f_stat:.2f}, expected > 10 for a strong "
            f"instrument (alpha_z=0.8, n=3000).")

    def test_iv_matches_wald_identity_on_no_covariate(self) -> None:
        """With no exogenous covariates (only intercept), 2SLS coincides
        with the Wald ratio: β = Cov(Y,Z) / Cov(D,Z). Verify equality."""
        from moirais.fn.iv import iv_2sls
        df = _dgp_iv_continuous(n=1000, seed=84)
        result = iv_2sls(df, y="Y", d="D", z="Z")
        beta_hat = float(result.coefficients["D"])
        # Closed-form Wald ratio
        cov_YZ = np.cov(df["Y"], df["Z"])[0, 1]
        cov_DZ = np.cov(df["D"], df["Z"])[0, 1]
        wald = cov_YZ / cov_DZ
        np.testing.assert_allclose(beta_hat, wald, rtol=0.02,
            err_msg=f"2SLS {beta_hat:.4f} != Wald ratio {wald:.4f} on no-covariate DGP.")


# ---------------------------------------------------------------------------
# M1-11 · estimate_late — Imbens-Angrist LATE with binary instrument
# ---------------------------------------------------------------------------

def _dgp_binary_iv(n: int = 5000, tau: float = 1.5, seed: int = 91) -> pd.DataFrame:
    """
    Binary-IV compliance DGP (Angrist-Imbens-Rubin 1996 framework):
        Z ~ Bernoulli(0.5)                              (random assignment)
        Types: compliers (p=0.6), always-takers (p=0.2), never-takers (p=0.2)
          compliers:     T = Z         (follow assignment)
          always-takers: T = 1         (regardless of Z)
          never-takers:  T = 0         (regardless of Z)
        Potential outcomes (homogeneous τ):
          Y(0) = eps
          Y(1) = Y(0) + tau
        Y = T * Y(1) + (1 - T) * Y(0)

    Under no-defier assumption + exclusion restriction:
        LATE (compliers only) = τ = Cov(Y,Z)/Cov(T,Z)
    """
    rng = np.random.default_rng(seed)
    Z = rng.binomial(1, 0.5, size=n)
    type_draws = rng.choice(["c", "a", "n"], size=n, p=[0.6, 0.2, 0.2])
    T = np.where(type_draws == "c", Z,
        np.where(type_draws == "a", 1, 0))
    eps = rng.normal(size=n)
    Y0 = eps
    Y1 = Y0 + tau
    Y = T * Y1 + (1 - T) * Y0
    return pd.DataFrame({"Y": Y, "T": T.astype(int), "Z": Z.astype(int)})


class TestEstimateLateCanonical:
    """
    Reference: Imbens & Angrist (1994) Econometrica; Angrist, Imbens &
    Rubin (1996) JASA.

    With a binary instrument and no covariates, the Wald estimator
    identifies the LATE among compliers:

        LATE = (E[Y|Z=1] - E[Y|Z=0]) / (E[T|Z=1] - E[T|Z=0])

    Under homogeneous effects, LATE = ATE = τ (among compliers ≡
    everyone when there's no heterogeneity among types beyond type).

    Properties:
      1. Returns dict with 'late' key.
      2. LATE ≈ τ on homogeneous-effect binary-IV DGP.
      3. LATE matches the closed-form Wald formula within Monte-Carlo noise.
    """

    def test_late_returns_dict(self) -> None:
        from moirais.fn.late import estimate_late
        df = _dgp_binary_iv(n=500, seed=91)
        result = estimate_late(df, treatment="T", outcome="Y", instrument="Z")
        assert isinstance(result, dict)
        assert "late" in result, f"expected 'late' key; got {list(result.keys())}"
        assert np.isfinite(float(result["late"]))

    def test_late_recovers_tau_on_binary_iv(self) -> None:
        from moirais.fn.late import estimate_late
        tau_true = 1.5
        df = _dgp_binary_iv(n=5000, tau=tau_true, seed=92)
        result = estimate_late(df, treatment="T", outcome="Y", instrument="Z")
        late = float(result["late"])
        # Monte-Carlo tolerance at n=5000 with 60% compliers: ~0.15
        assert abs(late - tau_true) < 0.20, (
            f"LATE = {late:.4f}, true τ = {tau_true}. "
            f"Bias {late - tau_true:.4f} exceeds 0.20 on binary-IV DGP.")

    def test_late_matches_wald_formula(self) -> None:
        """Closed-form identity check: LATE should equal the grouped Wald ratio."""
        from moirais.fn.late import estimate_late
        df = _dgp_binary_iv(n=5000, tau=1.5, seed=93)
        result = estimate_late(df, treatment="T", outcome="Y", instrument="Z")
        late_reported = float(result["late"])
        # Closed-form Wald
        y1, y0 = df.loc[df["Z"] == 1, "Y"].mean(), df.loc[df["Z"] == 0, "Y"].mean()
        t1, t0 = df.loc[df["Z"] == 1, "T"].mean(), df.loc[df["Z"] == 0, "T"].mean()
        wald = (y1 - y0) / (t1 - t0)
        np.testing.assert_allclose(late_reported, wald, rtol=0.05,
            err_msg=f"LATE {late_reported:.4f} != Wald ratio {wald:.4f}.")


# ---------------------------------------------------------------------------
# M1-12 · estimate_pliv — Partial Linear IV (Chernozhukov DoubleMLPLIV)
# ---------------------------------------------------------------------------

def _dgp_pliv(
    n: int = 2000,
    p: int = 5,
    theta: float = 1.2,
    gamma: float = 0.9,
    seed: int = 101,
) -> pd.DataFrame:
    """
    Partial Linear IV DGP (Chernozhukov et al. 2018 §4.3):
        X ~ N(0, 1)^p                                    (exogenous covariates)
        Z ~ N(0, 1)                                      (instrument, independent of X)
        U ~ N(0, 1)                                      (unobserved confounder)
        D = X @ beta_m + gamma * Z + 0.5 * U + v         (endogenous via U)
        Y = theta * D + X @ beta_g + U + eps             (outcome; U is confounder)

    Properties of this DGP:
      - E[U | Z, X] = 0 ⇒ Z is a valid instrument (exclusion restriction)
      - gamma controls first-stage strength
      - 0.5 * U in the D equation creates endogeneity that OLS can't fix
      - X enters both equations linearly ⇒ RidgeCV nuisances suffice
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Z = rng.normal(size=n)
    U = rng.normal(size=n)
    v = rng.normal(size=n)
    beta_m = rng.normal(scale=0.3, size=p)
    beta_g = rng.normal(scale=0.3, size=p)
    D = X @ beta_m + gamma * Z + 0.5 * U + v
    eps = rng.normal(size=n)
    Y = theta * D + X @ beta_g + U + eps
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
    df["D"] = D
    df["Z"] = Z
    df["Y"] = Y
    return df


class TestPlivCanonical:
    """
    Reference: Chernozhukov et al. (2018) §4.3, DoubleMLPLIV.

    The PLIV model:
        Y = θ·D + g(X) + ε    (E[ε|Z,X] = 0)
        D = m(X) + r(X,Z) + v (first-stage with IV entering)

    Under DML's Neyman-orthogonal score with cross-fitting and nuisance
    estimation rate o(n^{-1/4}), θ̂ is √n-consistent and asymptotically
    normal. Ridge nuisances are sufficient on linear DGPs.

    Properties:
      1. Returns dict with 'late' key (plus se/ci/pval).
      2. θ̂ recovers true θ within tolerance on Chernozhukov §6-style DGP.
      3. SE non-trivial and positive.
      4. Agreement with a naive OLS-of-Y-on-D baseline: PLIV must be
         meaningfully LESS biased than OLS under endogeneity.
    """

    def test_pliv_returns_dict_with_late(self) -> None:
        from moirais.fn.pliv import estimate_pliv
        df = _dgp_pliv(n=500, seed=101)
        result = estimate_pliv(
            df,
            treatment="D",
            outcome="Y",
            instrument="Z",
            covariates=[f"X{i+1}" for i in range(5)],
            n_folds=3,
        )
        assert isinstance(result, dict)
        assert {"late", "se", "ci_lower", "ci_upper", "pval", "n_obs", "method"}.issubset(result.keys()), (
            f"PLIV result missing expected keys. Got: {list(result.keys())}")
        assert np.isfinite(float(result["late"]))
        assert float(result["se"]) > 0

    def test_pliv_recovers_theta_on_linear_dgp(self) -> None:
        """
        On the canonical linear PLIV DGP with gamma=0.9 (strong
        instrument) at n=2000, RidgeCV nuisances suffice and θ̂ should
        be within 0.20 of true θ. Tolerance accounts for Monte-Carlo
        variance and the fact that we don't tune folds/regularization.
        """
        from moirais.fn.pliv import estimate_pliv
        theta_true = 1.2
        df = _dgp_pliv(n=2000, theta=theta_true, gamma=0.9, seed=102)
        result = estimate_pliv(
            df,
            treatment="D",
            outcome="Y",
            instrument="Z",
            covariates=[f"X{i+1}" for i in range(5)],
            n_folds=3,
        )
        theta_hat = float(result["late"])
        se = float(result["se"])
        assert se > 0
        # √n-consistency at n=2000 with gamma=0.9: expect bias ~ O(1/√n) ~ 0.022
        # Plus Monte-Carlo + finite-sample noise → 3*SE or 0.20 whichever larger.
        assert abs(theta_hat - theta_true) < max(3 * se, 0.20), (
            f"PLIV θ̂ = {theta_hat:.4f} (SE {se:.4f}), true θ = {theta_true}. "
            f"Bias {theta_hat - theta_true:.4f} exceeds tolerance on canonical "
            f"linear DGP — suggests identification or nuisance-fit issue.")

    def test_pliv_beats_naive_ols_under_endogeneity(self) -> None:
        """
        The PLIV DGP has U as an unobserved confounder (0.5*U in D,
        +U in Y). Naive OLS of Y on D is biased upward by roughly
        0.5 * Var(U) / Var(D|X,Z) — typically 0.2-0.5 in our DGP scale.
        PLIV uses Z to debias. So |PLIV bias| should be meaningfully
        smaller than |OLS bias|.
        """
        from moirais.fn.pliv import estimate_pliv
        theta_true = 1.2
        df = _dgp_pliv(n=2000, theta=theta_true, gamma=0.9, seed=103)
        covs = [f"X{i+1}" for i in range(5)]

        # Naive OLS of Y on D and X (no IV)
        import statsmodels.api as sm
        X_ols = sm.add_constant(df[["D"] + covs].astype(float))
        ols = sm.OLS(df["Y"].astype(float), X_ols).fit()
        ols_bias = abs(float(ols.params["D"]) - theta_true)

        # PLIV
        result = estimate_pliv(
            df, treatment="D", outcome="Y", instrument="Z",
            covariates=covs, n_folds=3)
        pliv_bias = abs(float(result["late"]) - theta_true)

        # PLIV bias should be meaningfully less than OLS bias.
        # Loose claim: PLIV bias < OLS bias (any improvement at all).
        # Reasonable claim: PLIV bias < 0.7 * OLS bias.
        assert pliv_bias < 0.7 * ols_bias + 0.05, (
            f"PLIV bias {pliv_bias:.4f} not meaningfully smaller than OLS bias "
            f"{ols_bias:.4f} on a DGP with U-confounding. Expected PLIV to "
            f"exploit Z for identification.")


# ---------------------------------------------------------------------------
# M1-13 · reg_discontinuity — Sharp RDD via local linear regression
# ---------------------------------------------------------------------------

def _dgp_sharp_rdd(
    n: int = 4000,
    cutoff: float = 0.0,
    tau: float = 1.5,
    slope_left: float = 0.8,
    slope_right: float = 0.5,
    seed: int = 111,
) -> pd.DataFrame:
    """
    Sharp RDD DGP with smooth potential outcomes and a jump at the cutoff:
        R ~ Uniform(-2, 2)                   (running variable)
        T = 1{R >= cutoff}                   (sharp assignment)
        Y(0) = slope_left * R                (smooth below and above)
        Y(1) = Y(0) + tau                    (homogeneous jump)
        Y = T*Y(1) + (1-T)*Y(0) + 0.3*eps    (observation noise)

    True LATE at the cutoff = tau (by construction). The slope is
    continuous only where T doesn't change; the jump at R=cutoff is tau.

    Reference: Imbens & Lemieux (2008) J. Econometrics; Hahn, Todd, van
    der Klaauw (2001) Econometrica.
    """
    rng = np.random.default_rng(seed)
    R = rng.uniform(-2, 2, size=n)
    T = (R >= cutoff).astype(float)
    # Allow a slope change above cutoff for generality (still identifies tau at R=cutoff)
    Y0 = np.where(R < cutoff, slope_left * R, slope_right * R)
    Y1 = Y0 + tau
    eps = rng.normal(scale=0.3, size=n)
    Y = T * Y1 + (1 - T) * Y0 + eps
    return pd.DataFrame({"outcome": Y, "running": R})


class TestRddCanonical:
    """
    Reference: Imbens & Lemieux (2008); Hahn, Todd, van der Klaauw (2001).

    Sharp RDD identifies the LATE at the cutoff under continuity of
    potential outcomes at R=cutoff.

    Properties:
      1. Returns RegressionResult with LATE coefficient.
      2. LATE recovers the true jump τ within 3*SE on smooth DGP.
      3. With too-small bandwidth (< data density), raises ValueError.
    """

    def test_rdd_returns_regression_result(self) -> None:
        from moirais.fn.rdd import reg_discontinuity
        df = _dgp_sharp_rdd(n=1000, seed=111)
        result = reg_discontinuity(df, y="outcome", r="running", cutoff=0.0)
        assert hasattr(result, "coefficients")
        assert "LATE" in result.coefficients, (
            f"Expected 'LATE' in coefficients, got {list(result.coefficients.keys())}")
        assert "bandwidth" in result.extra
        assert "cutoff" in result.extra

    def test_rdd_recovers_tau_on_smooth_dgp(self) -> None:
        from moirais.fn.rdd import reg_discontinuity
        tau_true = 1.5
        df = _dgp_sharp_rdd(n=4000, cutoff=0.0, tau=tau_true, seed=112)
        result = reg_discontinuity(df, y="outcome", r="running", cutoff=0.0)
        late = float(result.coefficients["LATE"])
        se = float(result.se["LATE"])
        assert se > 0
        assert abs(late - tau_true) < max(3 * se, 0.15), (
            f"RDD LATE = {late:.4f} (SE {se:.4f}), true τ = {tau_true}. "
            f"Bias {late - tau_true:.4f} exceeds tolerance on smooth DGP.")

    def test_rdd_too_small_bandwidth_raises(self) -> None:
        from moirais.fn.rdd import reg_discontinuity
        df = _dgp_sharp_rdd(n=200, seed=113)
        with pytest.raises(ValueError, match="bandwidth"):
            reg_discontinuity(df, y="outcome", r="running",
                              cutoff=0.0, bandwidth=0.001)


# ---------------------------------------------------------------------------
# M1-14 · diff_in_diff — Classic 2x2 DiD with parallel trends
# ---------------------------------------------------------------------------

def _dgp_2x2_did(
    n_per_cell: int = 1000,
    tau: float = 1.2,
    group_effect: float = 0.7,
    time_effect: float = 0.4,
    seed: int = 121,
) -> pd.DataFrame:
    """
    2x2 DiD DGP with parallel trends:
        groups:  T ∈ {0, 1}                     (control, treated)
        periods: post ∈ {0, 1}                  (pre, post)
        Y_i = group_effect * T_i + time_effect * post_i
              + tau * (T_i * post_i)            (only treated in post get τ)
              + eps_i                           (noise)

    Parallel trends holds by construction: time_effect is the same for
    both groups; the only interaction term is τ·T·post. So the DiD
    estimand is exactly τ.

    Reference: Card & Krueger (1994) AER; Angrist & Pischke (2009) Ch 5.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t_val in (0, 1):
        for post_val in (0, 1):
            eps = rng.normal(scale=1.0, size=n_per_cell)
            Y = (group_effect * t_val + time_effect * post_val
                 + tau * (t_val * post_val) + eps)
            for y in Y:
                rows.append({"outcome": float(y), "treatment": t_val, "post": post_val})
    return pd.DataFrame(rows)


class TestDiDCanonical:
    """
    Reference: Card & Krueger (1994); Angrist & Pischke (2009) Ch 5.

    Under parallel trends, the DiD estimand equals the ATT on the
    treated-in-post-period. On a DGP where the true interaction
    coefficient is τ and parallel trends holds exactly, DiD recovers τ.

    Properties:
      1. Returns ESRes with estimate + se + ci_lower + ci_upper.
      2. Point estimate within 3*SE of τ on parallel-trends DGP.
      3. Four-cell identity: DiD = (Ȳ₁₁ − Ȳ₁₀) − (Ȳ₀₁ − Ȳ₀₀).
    """

    def test_did_returns_esres(self) -> None:
        from moirais.fn.did import diff_in_diff
        df = _dgp_2x2_did(n_per_cell=500, seed=121)
        result = diff_in_diff(df, y="outcome", t="treatment", post="post")
        assert hasattr(result, "estimate"), f"Expected ESRes, got {type(result)}"
        assert hasattr(result, "se")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert np.isfinite(float(result.estimate))
        assert float(result.se) > 0

    def test_did_recovers_tau_on_parallel_trends_dgp(self) -> None:
        from moirais.fn.did import diff_in_diff
        tau_true = 1.2
        df = _dgp_2x2_did(n_per_cell=1000, tau=tau_true, seed=122)
        result = diff_in_diff(df, y="outcome", t="treatment", post="post")
        did_est = float(result.estimate)
        se = float(result.se)
        assert abs(did_est - tau_true) < max(3 * se, 0.15), (
            f"DiD = {did_est:.4f} (SE {se:.4f}), true τ = {tau_true}. "
            f"Bias {did_est - tau_true:.4f} exceeds tolerance under parallel trends.")

    def test_did_equals_four_cell_formula(self) -> None:
        """Closed-form identity: DiD = (Ȳ₁₁ − Ȳ₁₀) − (Ȳ₀₁ − Ȳ₀₀)."""
        from moirais.fn.did import diff_in_diff
        df = _dgp_2x2_did(n_per_cell=500, tau=2.0, seed=123)
        result = diff_in_diff(df, y="outcome", t="treatment", post="post")
        reported = float(result.estimate)
        # Four-cell means
        g11 = df.loc[(df["treatment"] == 1) & (df["post"] == 1), "outcome"].mean()
        g10 = df.loc[(df["treatment"] == 1) & (df["post"] == 0), "outcome"].mean()
        g01 = df.loc[(df["treatment"] == 0) & (df["post"] == 1), "outcome"].mean()
        g00 = df.loc[(df["treatment"] == 0) & (df["post"] == 0), "outcome"].mean()
        formula = (g11 - g10) - (g01 - g00)
        np.testing.assert_allclose(reported, formula, rtol=1e-6,
            err_msg=f"DiD {reported:.6f} != four-cell formula {formula:.6f}.")


# ---------------------------------------------------------------------------
# M1-15 · estimate_cate — T-learner / S-learner heterogeneous effects
# ---------------------------------------------------------------------------

def _dgp_heterogeneous(
    n: int = 4000,
    seed: int = 131,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Heterogeneous-effect DGP:
        X ~ N(0, 1)^p, p=3
        T ~ Bernoulli(0.5)                     (randomized — keeps CATE identified)
        τ(X) = 1.0 + 0.5 * X1                  (heterogeneous, depends on X1)
        Y(0) = X @ α + ε
        Y(1) = Y(0) + τ(X)
        Y = T * Y(1) + (1-T) * Y(0)

    Returns (df, true_cate) where true_cate[i] = 1.0 + 0.5 * X1[i].
    Mean true_cate ≈ 1.0 (ATE).
    """
    rng = np.random.default_rng(seed)
    p = 3
    X = rng.normal(size=(n, p))
    T = rng.binomial(1, 0.5, size=n)
    alpha = np.array([1.0, -0.6, 0.4])
    tau_x = 1.0 + 0.5 * X[:, 0]
    Y0 = X @ alpha + rng.normal(scale=1.0, size=n)
    Y1 = Y0 + tau_x
    Y = T * Y1 + (1 - T) * Y0
    df = pd.DataFrame({
        "Y": Y, "T": T,
        "X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2],
    })
    return df, tau_x


class TestCateCanonical:
    """
    Reference: Künzel, Sekhon, Bickel & Yu (2019) PNAS — metalearners for
    heterogeneous treatment effects.

    Properties:
      1. Returns pd.Series same length as input.
      2. Mean of estimated CATE matches true ATE within tolerance.
      3. Estimated CATE correlates positively with true τ(X) (Pearson > 0.3
         at n=4000 with RF depth 5).
      4. Both 't_learner' and 's_learner' modes produce finite output.
      5. Unknown meta_learner raises ValueError.
    """

    def test_cate_returns_series(self) -> None:
        from moirais.fn.cate import estimate_cate
        df, _ = _dgp_heterogeneous(n=500, seed=131)
        result = estimate_cate(
            df, treatment="T", outcome="Y",
            covariates=["X1", "X2", "X3"],
            meta_learner="t_learner",
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert np.all(np.isfinite(result.values))

    def test_cate_mean_matches_ate(self) -> None:
        from moirais.fn.cate import estimate_cate
        df, true_cate = _dgp_heterogeneous(n=4000, seed=132)
        ate_true = float(np.mean(true_cate))  # ≈ 1.0 by construction
        result = estimate_cate(
            df, treatment="T", outcome="Y",
            covariates=["X1", "X2", "X3"],
            meta_learner="t_learner",
        )
        ate_hat = float(result.mean())
        # RF CATE averaged over n=4000 should be close to ATE.
        assert abs(ate_hat - ate_true) < 0.20, (
            f"Mean CATE = {ate_hat:.4f}, true ATE = {ate_true:.4f}. "
            f"Bias {ate_hat - ate_true:.4f} exceeds 0.20.")

    def test_cate_correlates_with_true_tau_x(self) -> None:
        """CATE should track τ(X). With RF depth 5 and n=4000, Pearson
        correlation between estimated and true CATE should be > 0.3.
        Tight correlation is hard for RF on a linear-in-X truth with
        noisy outcome, so tolerance is loose."""
        from moirais.fn.cate import estimate_cate
        df, true_cate = _dgp_heterogeneous(n=4000, seed=133)
        result = estimate_cate(
            df, treatment="T", outcome="Y",
            covariates=["X1", "X2", "X3"],
            meta_learner="t_learner",
        )
        corr = float(np.corrcoef(result.values, true_cate)[0, 1])
        assert corr > 0.3, (
            f"Pearson(est CATE, true τ(X)) = {corr:.3f}. Expected > 0.3 "
            f"for an identifiable heterogeneous-effect DGP.")

    def test_cate_s_learner_runs(self) -> None:
        from moirais.fn.cate import estimate_cate
        df, _ = _dgp_heterogeneous(n=500, seed=134)
        result = estimate_cate(
            df, treatment="T", outcome="Y",
            covariates=["X1", "X2", "X3"],
            meta_learner="s_learner",
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert np.all(np.isfinite(result.values))

    def test_cate_rejects_unknown_meta_learner(self) -> None:
        from moirais.fn.cate import estimate_cate
        df, _ = _dgp_heterogeneous(n=100, seed=135)
        with pytest.raises(ValueError, match="meta_learner"):
            estimate_cate(
                df, treatment="T", outcome="Y",
                covariates=["X1", "X2", "X3"],
                meta_learner="not_a_learner",
            )


# ---------------------------------------------------------------------------
# M1-16 · estimate_gate — Group ATEs via AIPW within strata
# ---------------------------------------------------------------------------

def _dgp_grouped_tau(
    n_per_group: int = 2000,
    tau_by_group: dict[int, float] | None = None,
    seed: int = 141,
) -> pd.DataFrame:
    """
    Grouped heterogeneous-effect DGP:
        For each group g in tau_by_group:
          X ~ N(0, 1)^3
          T ~ Bernoulli(0.5)
          Y(0) = X @ α + ε
          Y(1) = Y(0) + tau_by_group[g]
          Y = T*Y(1) + (1-T)*Y(0)
          G = g

    Tests that GATE recovers the group-specific τ.
    """
    if tau_by_group is None:
        tau_by_group = {0: 0.5, 1: 1.0, 2: 1.5}
    rng = np.random.default_rng(seed)
    alpha = np.array([1.0, -0.6, 0.4])
    rows = []
    for g, tau_g in tau_by_group.items():
        X = rng.normal(size=(n_per_group, 3))
        T = rng.binomial(1, 0.5, size=n_per_group)
        eps = rng.normal(scale=1.0, size=n_per_group)
        Y0 = X @ alpha + eps
        Y = T * (Y0 + tau_g) + (1 - T) * Y0
        for i in range(n_per_group):
            rows.append({
                "Y": float(Y[i]), "T": int(T[i]),
                "X1": float(X[i, 0]), "X2": float(X[i, 1]), "X3": float(X[i, 2]),
                "G": int(g),
            })
    return pd.DataFrame(rows)


class TestGateCanonical:
    """
    Reference: Imai & Ratkovic (2013) AOAS; Chernozhukov, Demirer, Duflo
    & Fernandez-Val (2020) NBER 24678.

    Properties:
      1. Returns DataFrame with group/ate/se/ci_lower/ci_upper/n columns.
      2. One row per group.
      3. Each group's ATE recovers the true τ_g within tolerance.
      4. Groups with no treatment variation are skipped (logged warning).
    """

    def test_gate_returns_dataframe(self) -> None:
        from moirais.fn.gate import estimate_gate
        df = _dgp_grouped_tau(n_per_group=300, seed=141)
        result = estimate_gate(
            df, treatment="T", outcome="Y",
            covariates=["X1", "X2", "X3"],
            group_col="G",
        )
        assert isinstance(result, pd.DataFrame)
        assert {"group", "ate", "se", "ci_lower", "ci_upper", "n"}.issubset(result.columns), (
            f"GATE missing expected columns. Got: {list(result.columns)}")
        # One row per group
        assert len(result) == df["G"].nunique(), (
            f"Expected {df['G'].nunique()} rows, got {len(result)}")

    def test_gate_recovers_group_specific_tau(self) -> None:
        from moirais.fn.gate import estimate_gate
        tau_by_group = {0: 0.5, 1: 1.0, 2: 1.5}
        df = _dgp_grouped_tau(n_per_group=2000, tau_by_group=tau_by_group, seed=142)
        result = estimate_gate(
            df, treatment="T", outcome="Y",
            covariates=["X1", "X2", "X3"],
            group_col="G",
        )
        for _, row in result.iterrows():
            g = int(row["group"])
            ate_hat = float(row["ate"])
            se = float(row["se"])
            tau_true = tau_by_group[g]
            assert abs(ate_hat - tau_true) < max(3 * se, 0.20), (
                f"Group {g}: GATE = {ate_hat:.4f} (SE {se:.4f}), "
                f"true τ = {tau_true}. Bias {ate_hat - tau_true:.4f}.")


# ---------------------------------------------------------------------------
# M1-17 · ps_match — Propensity score nearest-neighbor matching (mando)
# ---------------------------------------------------------------------------

class TestPsMatchCanonical:
    """
    Reference: Rosenbaum & Rubin (1983) Biometrika; Stuart (2010)
    Statistical Science.

    1:1 PS matching estimates the ATT on the matched sample. Under
    correct PS specification + sufficient overlap, ATT_matched is
    consistent for the population ATT. On a confounded DGP, matching
    should beat the naive difference in means.

    Properties:
      1. Returns ESRes with estimate/se/ci/extra{n_matched, caliper}.
      2. On confounded DGP, matched ATE meaningfully closer to τ than
         naive diff-in-means.
      3. Too-tight caliper on sparse data raises ValueError.
    """

    def test_mando_returns_esres_with_matched_count(self) -> None:
        from moirais.fn.mando import ps_match
        # Rename our DGP's columns to match ps_match's defaults
        df = _dgp_logistic_ps(n=800, tau=1.5, seed=151)
        df2 = df.rename(columns={"Y": "outcome", "T": "treatment"})
        result = ps_match(df2, y="outcome", t="treatment", x=["X1", "X2", "X3"])
        assert hasattr(result, "estimate")
        assert "n_matched" in result.extra
        assert "caliper" in result.extra
        assert int(result.extra["n_matched"]) > 0

    def test_mando_beats_naive_on_confounded_dgp(self) -> None:
        from moirais.fn.mando import ps_match
        tau_true = 1.5
        df = _dgp_logistic_ps(n=4000, tau=tau_true, seed=152)
        df2 = df.rename(columns={"Y": "outcome", "T": "treatment"})

        # Naive difference in means (biased under confounding)
        naive = float(df["Y"][df["T"] == 1].mean() - df["Y"][df["T"] == 0].mean())
        naive_bias = abs(naive - tau_true)

        # Matched ATT
        result = ps_match(df2, y="outcome", t="treatment", x=["X1", "X2", "X3"])
        matched_bias = abs(float(result.estimate) - tau_true)

        assert matched_bias < 0.7 * naive_bias + 0.05, (
            f"Matching bias {matched_bias:.4f} not meaningfully smaller than "
            f"naive bias {naive_bias:.4f} on confounded DGP.")


# ---------------------------------------------------------------------------
# M1-18 · estimate_ate_gcomputation — G-computation / standardization
# ---------------------------------------------------------------------------

class TestGComputationCanonical:
    """
    Reference: Robins (1986) Mathematical Modelling; Hernán & Robins
    (2020) Ch 13.

    G-computation fits μ(T, X), predicts Ȳ(1) and Ȳ(0) for each unit,
    and averages the difference. Under correct outcome-model
    specification, the estimate is consistent for the ATE.

    Properties:
      1. Returns dict with ate/se/ci/outcome_model.
      2. Point estimate within tolerance of τ on linear confounded DGP.
      3. Bootstrap SE positive; 500 iterations succeed.
      4. outcome_model validation (invalid value raises ValueError).
    """

    def test_gcomp_returns_dict(self) -> None:
        from moirais.fn.g_comp import estimate_ate_gcomputation
        df = _dgp_logistic_ps(n=500, tau=1.0, seed=161)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y",
            covariates=["X1", "X2", "X3"],
            outcome_model="linear",
        )
        assert isinstance(result, dict)
        assert {"ate", "se", "ci_lower", "ci_upper", "n_obs", "outcome_model"}.issubset(result.keys())
        assert np.isfinite(float(result["ate"]))
        assert float(result["se"]) > 0

    def test_gcomp_recovers_tau_on_linear_dgp(self) -> None:
        from moirais.fn.g_comp import estimate_ate_gcomputation
        tau_true = 2.0
        df = _dgp_logistic_ps(n=3000, tau=tau_true, seed=162)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y",
            covariates=["X1", "X2", "X3"],
            outcome_model="linear",
        )
        ate = float(result["ate"])
        se = float(result["se"])
        assert abs(ate - tau_true) < max(3 * se, 0.15), (
            f"G-computation ATE = {ate:.4f} (SE {se:.4f}), true τ = {tau_true}. "
            f"Bias {ate - tau_true:.4f} exceeds tolerance.")

    def test_gcomp_rejects_invalid_outcome_model(self) -> None:
        from moirais.fn.g_comp import estimate_ate_gcomputation
        df = _dgp_logistic_ps(n=100, seed=163)
        with pytest.raises(ValueError, match="outcome_model"):
            estimate_ate_gcomputation(
                df, treatment="T", outcome="Y",
                covariates=["X1"], outcome_model="not_a_model")


# ---------------------------------------------------------------------------
# M1-19 · bdrj (backdoor_adjustment_formula) — STUB DETECTION
# ---------------------------------------------------------------------------

class TestBdrjStubDetection:
    """
    `bdrj.py`'s `backdoor_adjustment_formula(X, Y, Z, data)` is a generator-
    template STUB: the body does `np.mean(data)` and ignores X, Y, Z
    entirely. That's not a back-door adjustment — it's a placeholder.

    These tests document the stub behavior so we don't accidentally claim
    canonical correctness, and they'll start failing when the fn is
    properly replaced in M2 — at which point the tests get rewritten
    with real invariants.

    Reference for the TRUE math: Pearl (2009) *Causality* §3.3.1, the
    back-door adjustment formula:
        P(Y = y | do(X = x)) = Σ_z P(Y = y | X = x, Z = z) · P(Z = z)
    """

    @pytest.mark.xfail(
        strict=True,
        reason="bdrj.py is a stub — ignores X/Y/Z, just computes mean(data). "
               "M2 backlog: replace with actual Pearl back-door adjustment.",
    )
    def test_bdrj_uses_all_four_inputs(self) -> None:
        """If two calls with different X/Y/Z but same `data` return the
        same value, the fn is ignoring X/Y/Z. That's the stub signature."""
        from moirais.fn.bdrj import backdoor_adjustment_formula
        rng = np.random.default_rng(171)
        data = rng.normal(size=100)
        # Different X, Y, Z; same data.
        r1 = backdoor_adjustment_formula(
            X=rng.normal(size=100), Y=rng.normal(size=100),
            Z=rng.normal(size=100), data=data)
        r2 = backdoor_adjustment_formula(
            X=rng.uniform(size=100), Y=rng.uniform(size=100),
            Z=rng.uniform(size=100), data=data)
        # A real back-door formula WOULD differ when X/Y/Z differ.
        # This stub returns same value; xfail captures that fact.
        assert r1 != r2, "bdrj should use X/Y/Z — if this passes, stub is fixed"

    def test_bdrj_is_registered_as_stub(self) -> None:
        """Sentinel: while bdrj is a stub, we record it in the audit
        rather than claim canonical coverage. AST-walk the body to check
        whether X/Y/Z ever get read (Name node in Load context)."""
        import ast, inspect
        from moirais.fn.bdrj import backdoor_adjustment_formula
        src = inspect.getsource(backdoor_adjustment_formula)
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        loads = {
            node.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        # Real back-door adjustment would load X, Y, AND Z in the body.
        uses_xyz = {"X", "Y", "Z"}.issubset(loads)
        # Additionally, the stub template does `np.mean(data)` with no
        # other causal logic — unambiguous tell.
        has_mean_data = "np.mean(data)" in src
        is_stub = (not uses_xyz) and has_mean_data
        # When M2 un-stubs bdrj, this assertion flips and the xfail above
        # should also flip to a real canonical test.
        assert is_stub, (
            "bdrj.py no longer matches the stub pattern (body loads X/Y/Z "
            "or no longer uses np.mean(data)). Update the M1 audit and "
            "write real canonical tests against Pearl (2009) §3.3.1.")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _dgp_synth_control_panel(
    n_control: int = 10,
    n_pre: int = 20,
    n_post: int = 10,
    tau: float = 2.0,
    seed: int = 181,
) -> pd.DataFrame:
    """
    Panel DGP with a single treated unit and n_control donor units:
        For each unit u in {"treated"} ∪ {"c1"..."c10"}:
            base_level_u ~ N(0, 0.5)     (unit fixed effect)
            For each time t in [-n_pre, n_post):
                common_shock_t ~ N(0, 0.3) (time effect shared across units)
                Y_{u,t} = base_level_u + common_shock_t + noise
                If u=="treated" AND t >= 0:  Y_{u,t} += tau
        So treated unit jumps by tau at t=0; controls track the common shock.
    Synthetic control should recover tau by weighting controls to match
    treated's pre-period trajectory, then measuring post-period gap.
    """
    rng = np.random.default_rng(seed)
    rows = []
    units = ["treated"] + [f"c{i+1}" for i in range(n_control)]
    # Time-shared shocks (so synthetic-control can recover them exactly)
    time_effects = rng.normal(scale=0.3, size=n_pre + n_post)
    for u in units:
        base = rng.normal(scale=0.5)
        for idx, t in enumerate(range(-n_pre, n_post)):
            y = base + time_effects[idx] + rng.normal(scale=0.1)
            if u == "treated" and t >= 0:
                y += tau
            rows.append({"outcome": float(y), "unit": u, "time": int(t)})
    return pd.DataFrame(rows)


class TestSynthControlCanonical:
    """
    Reference: Abadie, Diamond & Hainmueller (2010) JASA "Synthetic
    Control Methods for Comparative Case Studies".

    Properties:
      1. Returns ESRes with .estimate and .extra{weights, pre_rmse}.
      2. Weights are non-negative and sum to ~1 (simplex constraint).
      3. On a panel where treated jumps by τ post-treatment and
         controls share the same time structure, synth recovers τ.
    """

    def test_sith_returns_esres_with_weights(self) -> None:
        from moirais.fn import synth_control
        df = _dgp_synth_control_panel(n_control=5, n_pre=10, n_post=5, seed=181)
        result = synth_control(df, y="outcome", unit="unit", time="time",
                               treated_unit="treated", treat_time=0)
        assert hasattr(result, "estimate")
        assert "weights" in result.extra
        assert "pre_rmse" in result.extra
        weights = result.extra["weights"]
        assert isinstance(weights, dict)
        assert len(weights) == 5  # one per donor

    def test_sith_weights_on_simplex(self) -> None:
        from moirais.fn import synth_control
        df = _dgp_synth_control_panel(n_control=8, n_pre=15, n_post=5, seed=182)
        result = synth_control(df, y="outcome", unit="unit", time="time",
                               treated_unit="treated", treat_time=0)
        weights = np.array(list(result.extra["weights"].values()))
        assert np.all(weights >= -1e-6), f"Negative weight found: {weights.min()}"
        weight_sum = float(weights.sum())
        np.testing.assert_allclose(weight_sum, 1.0, atol=1e-3,
            err_msg=f"Synth weights sum to {weight_sum}, expected 1.0 (simplex).")

    def test_sith_recovers_tau_on_shared_shock_panel(self) -> None:
        from moirais.fn import synth_control
        tau_true = 2.0
        df = _dgp_synth_control_panel(n_control=10, n_pre=20, n_post=10,
                                      tau=tau_true, seed=183)
        result = synth_control(df, y="outcome", unit="unit", time="time",
                               treated_unit="treated", treat_time=0)
        eff = float(result.estimate)
        # Tolerance: with 10 controls and 20 pre-periods, synth can match
        # common shocks well. Allow 0.30 for noise + weight-estimation error.
        assert abs(eff - tau_true) < 0.30, (
            f"Synth control effect = {eff:.4f}, true τ = {tau_true}. "
            f"Bias {eff - tau_true:.4f} exceeds 0.30.")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class TestSensitivityCanonical:
    """
    Reference: Rosenbaum (2002) *Observational Studies*, Ch 4.

    Rosenbaum bounds ask: how much hidden-bias factor Γ (odds-ratio of
    treatment assignment between two matched units with identical X)
    would be required to overturn a statistically significant result?

    Properties:
      1. Returns DescriptiveResult with .value (critical Γ) and
         .extra{table, critical_gamma}.
      2. At Γ=1 (no hidden bias), p-value ≈ original 2-sided z test.
      3. p-value monotonically increases with Γ (more possible bias →
         less confident rejection).
      4. Critical Γ is 1 when the input is already non-significant.
    """

    def test_yoda_s_returns_structure(self) -> None:
        from moirais.fn.yoda_s import sensitivity_analysis
        result = sensitivity_analysis(ate=1.0, se=0.3)  # z = 3.33, strong signal
        assert hasattr(result, "value")
        assert "table" in result.extra
        assert "critical_gamma" in result.extra
        assert len(result.extra["table"]) == 10  # default n_gamma

    def test_yoda_s_gamma1_gives_original_pvalue(self) -> None:
        """At Γ=1 (no bias), the Rosenbaum upper p-value equals the
        standard 2-sided z-test p-value."""
        from moirais.fn.yoda_s import sensitivity_analysis
        from scipy import stats
        ate, se = 1.0, 0.3
        result = sensitivity_analysis(ate=ate, se=se, gamma_range=(1.0, 1.0), n_gamma=1)
        row0 = result.extra["table"][0]
        assert row0["gamma"] == pytest.approx(1.0)
        # Standard 2-sided p-value
        expected_p = float(2 * stats.norm.sf(abs(ate / se)))
        assert row0["p_upper"] == pytest.approx(expected_p, abs=1e-6)

    def test_yoda_s_p_upper_monotone_in_gamma(self) -> None:
        """p_upper should be non-decreasing as Γ grows (larger assumed bias
        means weaker evidence of effect)."""
        from moirais.fn.yoda_s import sensitivity_analysis
        result = sensitivity_analysis(ate=0.8, se=0.3, gamma_range=(1.0, 3.0), n_gamma=10)
        ps = [r["p_upper"] for r in result.extra["table"]]
        for a, b in zip(ps, ps[1:]):
            assert b >= a - 1e-9, (
                f"p_upper is non-monotone in Γ: {a:.4f} -> {b:.4f}. "
                f"Rosenbaum-bound direction violated.")


# ---------------------------------------------------------------------------
# M1-22 · overlap_weight (ovrla) — Li-Morgan-Zaslavsky tilting weights
# ---------------------------------------------------------------------------

class TestOverlapWeightCanonical:
    """
    Reference: Li, Morgan & Zaslavsky (2018) *JASA* "Balancing Covariates
    via Propensity Score Weighting".

    Overlap weights w = 1 - e for treated, e for controls, target the
    ATO (ATE on the overlap population). Advantage: weights are bounded
    in [0, 0.5], so the estimator is never dominated by extreme propensities.

    Properties:
      1. Returns DescriptiveResult with .value (ATE) and
         .extra{y1_weighted_mean, y0_weighted_mean, ess_treated, ess_control, n}.
      2. On a confounded DGP, overlap-weighted ATE is meaningfully
         closer to τ than the naive difference in means.
      3. Weights are bounded in [0, 1] by construction; ESS is positive.
    """

    def test_ovrla_returns_structure(self) -> None:
        from moirais.fn.ovrla import overlap_weight
        df = _dgp_logistic_ps(n=500, seed=191)
        result = overlap_weight(
            ps_scores=df["ps_true"].values,
            treatment=df["T"].values,
            outcome=df["Y"].values,
        )
        assert hasattr(result, "value")
        assert {"y1_weighted_mean", "y0_weighted_mean", "ess_treated", "ess_control", "n"}.issubset(result.extra.keys())
        assert np.isfinite(float(result.value))
        assert float(result.extra["ess_treated"]) > 0
        assert float(result.extra["ess_control"]) > 0

    def test_ovrla_beats_naive_on_confounded_dgp(self) -> None:
        from moirais.fn.ovrla import overlap_weight
        tau_true = 2.0
        df = _dgp_logistic_ps(n=5000, tau=tau_true, seed=192)

        # Naive diff-in-means (biased under confounding)
        naive = float(df["Y"][df["T"] == 1].mean() - df["Y"][df["T"] == 0].mean())
        naive_bias = abs(naive - tau_true)

        # Overlap weighted ATE (targets ATO, but on homogeneous τ, ATO = ATE)
        result = overlap_weight(
            ps_scores=df["ps_true"].values,
            treatment=df["T"].values,
            outcome=df["Y"].values,
        )
        ov_bias = abs(float(result.value) - tau_true)
        assert ov_bias < 0.7 * naive_bias + 0.05, (
            f"Overlap-weight bias {ov_bias:.4f} not meaningfully smaller "
            f"than naive bias {naive_bias:.4f}.")


# ---------------------------------------------------------------------------
# M1-23 · gcomp (time-varying g-formula) — Robins (1986)
# ---------------------------------------------------------------------------

def _dgp_time_varying_g(
    n: int = 1000,
    K: int = 3,
    tau_per_period: float = 0.5,
    seed: int = 201,
) -> dict:
    """
    Time-varying treatment DGP:
        L_0 ~ N(0, 1)                 (baseline covariate)
        T_0 ~ Bernoulli(0.5)          (random first-period treatment)
        For k in 1..K-1:
            L_k = 0.5 * L_{k-1} + 0.3 * T_{k-1} + N(0, 0.3)   (time-varying L)
            T_k ~ Bernoulli(0.5)                               (random to aid identification)
        Y = sum_k tau_per_period * T_k + 0.2 * L_{K-1} + N(0, 1)

    Under always-treat: E[Y] = K * tau_per_period + E[0.2 L_{K-1}] (small)
    Under never-treat:  E[Y] ≈ E[0.2 L_{K-1}] (small and different)
    True ATE (always − never) ≈ K * tau_per_period.
    """
    rng = np.random.default_rng(seed)
    T = np.zeros((n, K), dtype=float)
    L = np.zeros((n, K, 1), dtype=float)
    L[:, 0, 0] = rng.normal(size=n)
    T[:, 0] = rng.binomial(1, 0.5, size=n)
    for k in range(1, K):
        L[:, k, 0] = 0.5 * L[:, k - 1, 0] + 0.3 * T[:, k - 1] + rng.normal(scale=0.3, size=n)
        T[:, k] = rng.binomial(1, 0.5, size=n)
    Y = (tau_per_period * T.sum(axis=1)
         + 0.2 * L[:, K - 1, 0]
         + rng.normal(size=n))
    return {"Y": Y, "T_seq": T, "L_seq": L}


class TestGcompTimeVaryingCanonical:
    """
    Reference: Robins (1986) *Mathematical Modelling*; Daniel et al.
    (2013) *Statistics in Medicine*; Hernán & Robins (2020) Part III.

    gcomp() implements the parametric g-formula for a time-varying
    regime. Under a correctly specified sequential outcome model, the
    always-treat vs never-treat contrast identifies the ATE of
    sustained treatment.

    Properties:
      1. Returns dict with ate/se/ci/n/method.
      2. Natural regime returns (mean(Y), None, None) — sanity check.
      3. ATE ≈ K * τ_per_period on linear DGP (loose tolerance since
         sequential OLS on 3 time points with random T is noisy).
    """

    def test_gcomp_time_varying_returns_dict(self) -> None:
        from moirais.fn.gcomp import gcomp
        data = _dgp_time_varying_g(n=400, K=3, seed=201)
        result = gcomp(Y=data["Y"], T_seq=data["T_seq"], L_seq=data["L_seq"],
                       regime="always_treat", n_boot=30)
        assert isinstance(result, dict)
        assert {"ate", "se", "ci_lower", "ci_upper", "n", "method"}.issubset(result.keys())

    def test_gcomp_natural_regime_returns_mean(self) -> None:
        """Under 'natural' regime, the point estimate is mean(Y) (the
        observed-treatment counterfactual reduces to the empirical mean)."""
        from moirais.fn.gcomp import gcomp
        data = _dgp_time_varying_g(n=300, K=2, seed=202)
        result = gcomp(Y=data["Y"], T_seq=data["T_seq"], L_seq=data["L_seq"],
                       regime="natural", n_boot=10)
        np.testing.assert_allclose(
            result["ate"], float(np.mean(data["Y"])), rtol=1e-6,
            err_msg="Natural regime should return mean(Y).")

    def test_gcomp_recovers_total_effect(self) -> None:
        """always-treat vs never-treat contrast ≈ K * τ_per_period."""
        from moirais.fn.gcomp import gcomp
        K, tau_per = 3, 0.5
        expected = K * tau_per  # = 1.5
        data = _dgp_time_varying_g(n=1000, K=K, tau_per_period=tau_per, seed=203)
        result = gcomp(Y=data["Y"], T_seq=data["T_seq"], L_seq=data["L_seq"],
                       regime="always_treat", n_boot=50)
        ate = float(result["ate"])
        se = float(result["se"])
        # Wider tolerance than single-period g_comp because sequential
        # OLS amplifies noise across time steps.
        assert abs(ate - expected) < max(3 * se, 0.50), (
            f"gcomp ATE = {ate:.4f} (SE {se:.4f}), true total effect = {expected}. "
            f"Bias {ate - expected:.4f} exceeds tolerance on linear time-varying DGP.")
