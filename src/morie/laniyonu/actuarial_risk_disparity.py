"""Reproducible replication of O'Connell & Laniyonu (2025) Race & Justice 15(3):428–453.

"Race, Gender, and Risk Assessments in Canadian Federal Prison" — an
audit of the Correctional Service of Canada's actuarial risk
instruments (Static, DFIA-R Dynamic, Offender Security Level,
Reintegration Potential) and the downstream binary outcomes (parole
granted; institutional housing level) over the 2011-2018 CSC FOI
release (n = 49,168 individuals, 51,769 sentences).

Two distinct empirical pieces the paper reports, both surfaced here:

  1. **Stage 1 — risk-score disparity.**  For each of the four
     ordinal scores, fit a *threshold-specific* cumulative-logit
     regression of the score on race × gender (+ age + priors).  The
     proportional-odds-violating finding is the headline: race bias
     is concentrated at the low→medium cutoff, not medium→high.
     Indigenous men carry +0.59 log-odds at the low→medium threshold
     of Offender Security Level.

  2. **Stage 2 — downstream-outcome disparity, net of score.**  For
     the binary outcomes (parole granted, housing-level placement),
     run the two-stage score-net-residual audit: regress the outcome
     on the actuarial score plus race × gender.  A non-zero residual
     race coefficient means CSC staff are NOT applying the score
     uniformly across race × gender cells.  The paper reports −11pp
     parole for Black & Indigenous men with the *best* reintegration
     score versus White men, and −22pp for Black women.

This module is the thin user-facing wrapper.  Heavy lifting lives in
two :mod:`morie.mrm_primitives` callables:

  - :func:`~morie.mrm_primitives.threshold_specific_ordinal`
  - :func:`~morie.mrm_primitives.score_net_residual`

We deliberately do not re-implement either primitive.  Callers with
data already in canonical form can compose the primitives directly;
this wrapper exists for the "I have the CSC-shaped frame" caller.

Caveat surfaced via :class:`UserWarning` (per Goel et al. 2021): a
non-zero residual race coefficient is evidence of *output* disparity,
not *predictive-validity* disparity.  The two are conceptually
distinct and the paper is explicit that staff-overrides and
disparate-treatment claims rest on the former, not the latter.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..mrm_primitives.ordinal import (
    ThresholdSpecificOrdinalResult,
    threshold_specific_ordinal,
)
from ..mrm_primitives.score_net_residual import (
    ScoreNetResidualResult,
    score_net_residual,
)


# Canonical outcome registry — maps user-facing outcome label to
# (kind, default-column-name).  Kept here so callers can override the
# column name without re-typing the "is this ordinal or binary" logic.
_ORDINAL_OUTCOMES: dict[str, str] = {
    "static": "static_score",
    "dynamic": "dynamic_score",
    "osl": "offender_security_level",
    "reintegration": "reintegration_potential",
}
_BINARY_OUTCOMES: dict[str, str] = {
    "parole": "parole_granted",
    "housing": "housing_level",
}


@dataclass
class ActuarialRiskDisparityResult:
    """Per-outcome disparity audit + diagnostics.

    Exactly one of ``ordinal_result`` / ``residual_result`` is populated,
    depending on whether ``outcome`` is one of the four ordinal risk
    scores or one of the two binary downstream outcomes.

    The ``per_threshold_logodds`` field is a flat per-stratum dict of
    the form ``{(gender_stratum, threshold_label, race_column): logodds}``,
    populated for stage-1 (ordinal) runs.  When ``split_by_gender=False``
    the gender stratum is ``"pooled"``.
    """
    outcome: str
    outcome_kind: Literal["ordinal", "binary"]
    outcome_col: str
    n_obs: int
    race_cols: list[str]
    gender_col: str
    ordinal_result: dict[str, ThresholdSpecificOrdinalResult] | None = None
    residual_result: ScoreNetResidualResult | None = None
    per_threshold_logodds: dict[tuple[str, str, str], float] = field(
        default_factory=dict
    )
    proportional_odds_lr_stat: float | None = None
    proportional_odds_lr_df: int | None = None
    proportional_odds_p: float | None = None
    note: str = ""

    def residual_race_coefficient(self, race_col: str) -> float | None:
        """Stage-2 residual race coefficient (None when stage 1)."""
        if self.residual_result is None:
            return None
        return self.residual_result.coefficients.get(race_col)

    def bias_concentrated_at_low_medium(
        self, race_col: str, gender_stratum: str = "pooled",
        ratio_threshold: float = 1.5,
    ) -> bool | None:
        """True iff |β_threshold1| / |β_threshold2| >= ratio_threshold.

        Returns None when not a stage-1 result, or when the second
        threshold's coefficient is too small to take the ratio safely.
        """
        if self.ordinal_result is None:
            return None
        res = self.ordinal_result.get(gender_stratum)
        if res is None or race_col not in res.covariate_names:
            return None
        i = res.covariate_names.index(race_col)
        if res.coefficients.shape[0] < 2:
            return None
        b1, b2 = res.coefficients[0, i], res.coefficients[1, i]
        if abs(b2) < 1e-6:
            return None
        return abs(b1) / abs(b2) >= ratio_threshold

    def interpret(self) -> str:
        head = (
            f"O'Connell & Laniyonu (2025) replication — "
            f"outcome={self.outcome!r} ({self.outcome_kind}), "
            f"col={self.outcome_col!r}, n={self.n_obs}."
        )
        lines = [head]
        if self.ordinal_result is not None:
            for stratum, res in self.ordinal_result.items():
                lines.append(f"  [{stratum}] " + res.interpret().replace("\n", "\n    "))
                # Headline pattern: bias concentrated at low→medium
                for rc in self.race_cols:
                    flag = self.bias_concentrated_at_low_medium(rc, stratum)
                    if flag is True:
                        b = res.coefficients[0, res.covariate_names.index(rc)]
                        lines.append(
                            f"    -> {rc}: bias concentrated at low->medium "
                            f"cutoff (β1={b:+.3f}); proportional-odds "
                            "assumption is empirically wrong here."
                        )
        if self.residual_result is not None:
            lines.append("  Stage-2 (score-net-residual): "
                         + self.residual_result.interpretation)
        if self.note:
            lines.append(f"  Note: {self.note}")
        # Output-vs-predictive-validity caveat, every time.
        lines.append(
            "  CAVEAT: this is OUTPUT disparity, not predictive-validity "
            "disparity.  See Goel et al. (2021) on why the two should "
            "not be conflated."
        )
        return "\n".join(lines)


def actuarial_risk_disparity(
    df: pd.DataFrame,
    *,
    outcome: Literal["static", "dynamic", "osl", "reintegration",
                     "parole", "housing"],
    race_cols: list[str],
    gender_col: str = "gender",
    score_col: str | None = None,
    control_cols: list[str] | None = None,
    ordinal_levels: list[str] | None = None,
    outcome_col: str | None = None,
    split_by_gender: bool = True,
    bootstrap_replicates: int = 200,
    random_state: int = 20260513,
) -> ActuarialRiskDisparityResult:
    """Audit CSC actuarial risk and downstream outcomes for race × gender bias.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level (one row per sentence) CSC microdata.  Race
        columns may be either pre-dummied 0/1 indicators or a single
        categorical race column — pass the indicator column names in
        ``race_cols`` (e.g. ``["race_indigenous", "race_black"]`` with
        White as the implicit reference).
    outcome : {"static", "dynamic", "osl", "reintegration",
               "parole", "housing"}
        Which audit to run.  The first four are the ordinal risk
        scores (stage 1 — threshold-specific ordinal logit).  The
        last two are the binary downstream outcomes (stage 2 — score-
        net-residual audit).
    race_cols : list[str]
        Race indicator columns.  Must be 0/1 numeric.  Reference
        category (typically White) is the omitted column.
    gender_col : str, default "gender"
        Categorical gender column.  Used either as a stratifying
        variable (when ``split_by_gender=True``) or one-hot-dummied
        and stacked into the design matrix (when False).
    score_col : str, optional
        Required when ``outcome`` is ``"parole"`` or ``"housing"`` —
        the actuarial score on which CSC claims to make the decision.
        Typically ``"reintegration_score_numeric"`` for parole and
        ``"osl_score_numeric"`` for housing.
    control_cols : list[str], optional
        Additional controls (age, prior offences, sentence length,
        offence-severity dummies, etc.).  Pre-dummy any categoricals.
    ordinal_levels : list[str], optional
        Override level ordering for ordinal outcomes.  Default
        ``["low", "medium", "high"]`` for all four risk scores —
        matches the CSC released form.  Ignored for binary outcomes.
    outcome_col : str, optional
        Override the default column name for the chosen ``outcome``.
        Defaults to the canonical name in the CSC FOI release
        (see :data:`_ORDINAL_OUTCOMES` / :data:`_BINARY_OUTCOMES`).
    split_by_gender : bool, default True
        Stage-1 only.  When True, fits the threshold-specific ordinal
        separately within each gender stratum — recovers the
        "Indigenous men +0.59" style estimate cleanly.  When False,
        adds a one-hot gender column to the pooled design.
    bootstrap_replicates : int, default 200
        Stage-2 only; bootstrap reps for residual race-coef SEs.
    random_state : int, default 20260513

    Returns
    -------
    ActuarialRiskDisparityResult

    Notes
    -----
    O'Connell & Laniyonu fit the Bayesian version; the primitive
    morie composes is the frequentist analogue (see
    :mod:`morie.mrm_primitives.ordinal` for the explicit trade-off).
    The threshold-specific coefficient *pattern* (β_low->med >>
    β_med->high for race) is recovered identically; only the
    posterior-credible-interval reporting is different.
    """
    if ordinal_levels is None:
        ordinal_levels = ["low", "medium", "high"]
    if control_cols is None:
        control_cols = []

    # The output-vs-predictive caveat is so load-bearing for this
    # paper that we surface it on EVERY call, not just in interpret().
    warnings.warn(
        "morie.laniyonu.actuarial_risk_disparity reports OUTPUT "
        "disparity, not predictive-validity disparity.  A non-zero "
        "residual race coefficient could reflect disparate treatment "
        "OR a valid race difference in the unobserved outcome — see "
        "Goel et al. (2021).  The paper is explicit that the disparate-"
        "treatment claim rests on output disparity given the policy "
        "framing of risk scores as binding inputs.",
        UserWarning,
        stacklevel=2,
    )

    # ── Resolve the outcome column ─────────────────────────────────
    if outcome in _ORDINAL_OUTCOMES:
        kind: Literal["ordinal", "binary"] = "ordinal"
        col = outcome_col or _ORDINAL_OUTCOMES[outcome]
    elif outcome in _BINARY_OUTCOMES:
        kind = "binary"
        col = outcome_col or _BINARY_OUTCOMES[outcome]
    else:
        raise ValueError(
            f"unknown outcome {outcome!r}; expected one of "
            f"{sorted(set(_ORDINAL_OUTCOMES) | set(_BINARY_OUTCOMES))}"
        )

    missing = [c for c in [col, gender_col, *race_cols, *control_cols]
               if c not in df.columns]
    if missing:
        raise KeyError(
            f"actuarial_risk_disparity: columns missing from df: {missing}"
        )

    # ── Stage 1: ordinal risk-score audit ──────────────────────────
    if kind == "ordinal":
        if score_col is not None:
            warnings.warn(
                "score_col is ignored for ordinal outcomes "
                f"({outcome!r}); the score IS the outcome here.",
                UserWarning,
                stacklevel=2,
            )
        return _run_ordinal_stage(
            df=df,
            outcome=outcome,
            outcome_col=col,
            race_cols=race_cols,
            gender_col=gender_col,
            control_cols=control_cols,
            ordinal_levels=ordinal_levels,
            split_by_gender=split_by_gender,
        )

    # ── Stage 2: binary downstream-outcome audit ───────────────────
    if score_col is None:
        raise ValueError(
            f"score_col is required for outcome={outcome!r}; pass the "
            "actuarial-score column name (e.g. "
            "'reintegration_score_numeric' for parole, "
            "'osl_score_numeric' for housing)."
        )
    return _run_residual_stage(
        df=df,
        outcome=outcome,
        outcome_col=col,
        score_col=score_col,
        race_cols=race_cols,
        gender_col=gender_col,
        control_cols=control_cols,
        split_by_gender=split_by_gender,
        bootstrap_replicates=bootstrap_replicates,
        random_state=random_state,
    )


# ─── stage 1 (ordinal) ─────────────────────────────────────────────


def _run_ordinal_stage(
    *,
    df: pd.DataFrame,
    outcome: str,
    outcome_col: str,
    race_cols: list[str],
    gender_col: str,
    control_cols: list[str],
    ordinal_levels: list[str],
    split_by_gender: bool,
) -> ActuarialRiskDisparityResult:
    """Run threshold-specific ordinal logit per gender stratum (or pooled)."""
    results: dict[str, ThresholdSpecificOrdinalResult] = {}
    per_thresh: dict[tuple[str, str, str], float] = {}

    if split_by_gender:
        gender_strata = sorted(df[gender_col].dropna().unique().tolist())
        for g in gender_strata:
            sub = df[df[gender_col] == g]
            if len(sub) < 30:
                warnings.warn(
                    f"gender stratum {g!r} has only {len(sub)} rows; "
                    "skipping (need n >= 30 for stable ordinal fit).",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            covariate_cols = [*race_cols, *control_cols]
            res = threshold_specific_ordinal(
                sub,
                outcome_col=outcome_col,
                covariate_cols=covariate_cols,
                ordinal_levels=ordinal_levels,
                fit_proportional_odds_first=True,
            )
            stratum_label = str(g)
            results[stratum_label] = res
            for k, thresh_label in enumerate(res.threshold_labels):
                for rc in race_cols:
                    if rc in res.covariate_names:
                        i = res.covariate_names.index(rc)
                        per_thresh[(stratum_label, thresh_label, rc)] = (
                            float(res.coefficients[k, i])
                        )
    else:
        # Pool genders, one-hot the gender column, fold into covariates.
        gender_dummies = pd.get_dummies(
            df[gender_col], prefix=gender_col, drop_first=True
        ).astype(float)
        df_aug = pd.concat([df, gender_dummies], axis=1)
        covariate_cols = [*race_cols, *gender_dummies.columns, *control_cols]
        res = threshold_specific_ordinal(
            df_aug,
            outcome_col=outcome_col,
            covariate_cols=covariate_cols,
            ordinal_levels=ordinal_levels,
            fit_proportional_odds_first=True,
        )
        results["pooled"] = res
        for k, thresh_label in enumerate(res.threshold_labels):
            for rc in race_cols:
                if rc in res.covariate_names:
                    i = res.covariate_names.index(rc)
                    per_thresh[("pooled", thresh_label, rc)] = (
                        float(res.coefficients[k, i])
                    )

    # PO-LR test summary: take the worst-p across strata as the
    # diagnostic (one rejection is enough to motivate threshold-specific).
    lr_stats = [(r.proportional_odds_lr_stat, r.proportional_odds_lr_df,
                 r.proportional_odds_lr_stat is not None
                 and r.proportional_odds_p is not None and r.proportional_odds_p)
                for r in results.values()
                if r.proportional_odds_lr_stat is not None]
    if lr_stats:
        # pick the stratum with the smallest p (strongest PO violation)
        worst = min(lr_stats, key=lambda t: t[2] if t[2] is not False else 1.0)
        worst_stat, worst_df, worst_p = worst
    else:
        worst_stat = worst_df = worst_p = None

    n_obs = int(sum(r.n_obs for r in results.values()))
    note = (
        "Stage 1 — threshold-specific ordinal logit.  Compare β at "
        "threshold 1 (low->medium) vs. threshold 2 (medium->high); a "
        "much larger |β_1| is the headline O'Connell-Laniyonu pattern."
    )
    return ActuarialRiskDisparityResult(
        outcome=outcome,
        outcome_kind="ordinal",
        outcome_col=outcome_col,
        n_obs=n_obs,
        race_cols=list(race_cols),
        gender_col=gender_col,
        ordinal_result=results,
        residual_result=None,
        per_threshold_logodds=per_thresh,
        proportional_odds_lr_stat=(float(worst_stat)
                                   if worst_stat is not None else None),
        proportional_odds_lr_df=(int(worst_df)
                                 if worst_df is not None else None),
        proportional_odds_p=(float(worst_p)
                             if worst_p not in (None, False) else None),
        note=note,
    )


# ─── stage 2 (binary, score-net-residual) ──────────────────────────


def _run_residual_stage(
    *,
    df: pd.DataFrame,
    outcome: str,
    outcome_col: str,
    score_col: str,
    race_cols: list[str],
    gender_col: str,
    control_cols: list[str],
    split_by_gender: bool,
    bootstrap_replicates: int,
    random_state: int,
) -> ActuarialRiskDisparityResult:
    """Run score_net_residual; gender either stratifies or stacks."""
    # For stage 2 we don't iterate per-stratum results into a dict —
    # the residual race coefficient is the headline quantity and the
    # paper reports race × gender INTERACTIONS, so we either:
    #   (a) stratify (split_by_gender=True): one residual race coef
    #       per gender stratum;
    #   (b) stack (split_by_gender=False): race + gender + race×gender.
    # Here we keep it simple and report (a) by concatenating per-
    # stratum residual race columns into a single result with
    # "<race>::<gender>" keys; (b) gets a flat additive design.
    if split_by_gender:
        gender_strata = sorted(df[gender_col].dropna().unique().tolist())
        combined_coefs: dict[str, float] = {}
        combined_ses: dict[str, float] = {}
        score_coefs: list[float] = []
        n_total = 0
        for g in gender_strata:
            sub = df[df[gender_col] == g]
            if len(sub) < 30:
                warnings.warn(
                    f"gender stratum {g!r} has only {len(sub)} rows; "
                    "skipping (need n >= 30 for stable bootstrap).",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            res_g = score_net_residual(
                sub,
                score_col=score_col,
                outcome_col=outcome_col,
                sensitive_cols=race_cols,
                control_cols=control_cols,
                family="logit",
                bootstrap_replicates=bootstrap_replicates,
                random_state=random_state,
            )
            for rc in race_cols:
                key = f"{rc}::{g}"
                combined_coefs[key] = res_g.coefficients.get(rc, float("nan"))
                combined_ses[key] = res_g.std_errors.get(rc, float("nan"))
            score_coefs.append(res_g.score_coefficient)
            n_total += res_g.n_obs

        interp = (
            f"Stratified by {gender_col}: " +
            ", ".join(f"{k}={v:+.4f}" for k, v in combined_coefs.items())
            + ".  Mean score coefficient across strata: "
            f"{(float(np.mean(score_coefs)) if score_coefs else float('nan')):+.4f}."
        )
        residual = ScoreNetResidualResult(
            sensitive_attribute_names=list(combined_coefs.keys()),
            coefficients=combined_coefs,
            std_errors=combined_ses,
            n_obs=n_total,
            score_coefficient=(float(np.mean(score_coefs))
                               if score_coefs else float("nan")),
            interpretation=interp,
        )
        n_obs = n_total
    else:
        gender_dummies = pd.get_dummies(
            df[gender_col], prefix=gender_col, drop_first=True
        ).astype(float)
        df_aug = pd.concat([df, gender_dummies], axis=1)
        sensitive_cols = [*race_cols, *gender_dummies.columns]
        residual = score_net_residual(
            df_aug,
            score_col=score_col,
            outcome_col=outcome_col,
            sensitive_cols=sensitive_cols,
            control_cols=control_cols,
            family="logit",
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state,
        )
        n_obs = residual.n_obs

    note = (
        "Stage 2 — score-net-residual.  Coefficient on each race "
        "indicator NET of the actuarial score is the disparate-"
        "treatment signal.  Paper's headline: Black & Indigenous men "
        "with the best reintegration score are still -11pp on parole "
        "vs. White; Black women -22pp."
    )
    return ActuarialRiskDisparityResult(
        outcome=outcome,
        outcome_kind="binary",
        outcome_col=outcome_col,
        n_obs=n_obs,
        race_cols=list(race_cols),
        gender_col=gender_col,
        ordinal_result=None,
        residual_result=residual,
        per_threshold_logodds={},
        note=note,
    )
