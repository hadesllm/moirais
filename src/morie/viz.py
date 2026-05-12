"""
Publication-ready statistical visualizations for epidemiological analysis.

Provides forest plots, funnel plots, survival curves, ROC curves, DAGs,
diagnostic plots, and many more -- all with a consistent theme system
suitable for journal submission.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------

_THEMES: dict[str, dict[str, Any]] = {
    "publication": {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6.5, 4.5),
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    },
    "presentation": {
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.figsize": (10, 7),
        "figure.dpi": 150,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    },
    "minimal": {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (5, 3.5),
        "figure.dpi": 150,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.grid": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
    },
}


def set_theme(name: str = "publication") -> None:
    """
    Apply a predefined MORIE matplotlib theme globally.

    Parameters
    ----------
    name : str
        One of ``"publication"``, ``"presentation"``, or ``"minimal"``.

    Raises
    ------
    ValueError
        If *name* is not a recognised theme.
    """
    if name not in _THEMES:
        raise ValueError(f"Unknown theme '{name}'. Choose from {list(_THEMES)}")
    plt.rcParams.update(_THEMES[name])
    logger.info("Applied '%s' theme", name)


def get_theme(name: str = "publication") -> dict[str, Any]:
    """Return a copy of the named theme dictionary."""
    if name not in _THEMES:
        raise ValueError(f"Unknown theme '{name}'. Choose from {list(_THEMES)}")
    return dict(_THEMES[name])


def _apply_theme_ctx(name: str | None):
    """Return a context manager that temporarily applies a theme."""
    if name is None:
        return plt.rc_context({})
    return plt.rc_context(_THEMES.get(name, _THEMES["publication"]))


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------


def forest_plot(
    labels: Sequence[str],
    estimates: Sequence[float],
    ci_lower: Sequence[float],
    ci_upper: Sequence[float],
    *,
    null_value: float = 0.0,
    xlabel: str = "Effect size",
    title: str = "",
    color: str = "#2c7bb6",
    diamond_summary: tuple[float, float, float] | None = None,
    weights: Sequence[float] | None = None,
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Forest plot for meta-analysis, subgroup effects, or regression coefficients.

    Parameters
    ----------
    labels : sequence of str
        Row labels (study names, subgroups, or variable names).
    estimates : sequence of float
        Point estimates.
    ci_lower, ci_upper : sequence of float
        Lower and upper bounds of the confidence interval.
    null_value : float
        The null/reference value; a vertical dashed line is drawn here.
    xlabel : str
        Label for the x-axis (e.g. ``"Odds Ratio"`` or ``"Beta"``).
    title : str
        Plot title.
    color : str
        Colour for the point estimates and CI lines.
    diamond_summary : tuple of (estimate, lower, upper) or None
        If provided, a diamond summarising the pooled estimate is drawn at
        the bottom.
    weights : sequence of float or None
        Optional study weights used to scale marker sizes.
    theme : str or None
        MORIE theme name.  Pass ``None`` to use current rc params.
    ax : matplotlib Axes or None
        Axes to draw on.  A new figure is created if ``None``.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    Lewis, S., & Clarke, M. (2001). Forest plots: trying to see the wood
    and the trees. *BMJ*, 322(7300), 1479-1480.
    https://doi.org/10.1136/bmj.322.7300.1479
    """
    labels = list(labels)
    estimates = np.asarray(estimates, dtype=float)
    ci_lower = np.asarray(ci_lower, dtype=float)
    ci_upper = np.asarray(ci_upper, dtype=float)
    n = len(labels)
    if not (len(estimates) == len(ci_lower) == len(ci_upper) == n):
        raise ValueError("labels, estimates, ci_lower, ci_upper must have same length")

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * n + 1)))
        else:
            fig = ax.figure

        y_positions = np.arange(n, 0, -1, dtype=float)

        # Marker sizes proportional to weights
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            sizes = 30 + 170 * (w / w.max())
        else:
            sizes = np.full(n, 80.0)

        # CI lines
        for i in range(n):
            ax.plot(
                [ci_lower[i], ci_upper[i]],
                [y_positions[i], y_positions[i]],
                color=color,
                linewidth=1.5,
                solid_capstyle="butt",
            )
        # Point estimates
        ax.scatter(estimates, y_positions, s=sizes, color=color, zorder=5, edgecolors="white", linewidths=0.5)

        # Null line
        ax.axvline(null_value, color="grey", linestyle="--", linewidth=0.8, zorder=1)

        # Summary diamond
        bottom_y = 0.0
        if diamond_summary is not None:
            de, dl, du = diamond_summary
            dy = -0.5
            bottom_y = dy - 0.8
            diamond_x = [dl, de, du, de]
            diamond_y = [dy, dy + 0.3, dy, dy - 0.3]
            ax.fill(diamond_x, diamond_y, color=color, alpha=0.6, zorder=5)
            ax.plot(diamond_x + [diamond_x[0]], diamond_y + [diamond_y[0]], color=color, linewidth=1)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
        ax.set_ylim(bottom_y if diamond_summary else 0.2, n + 0.8)
        ax.invert_yaxis()

        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Funnel plot
# ---------------------------------------------------------------------------


def funnel_plot(
    effects: Sequence[float],
    se: Sequence[float],
    *,
    null_value: float = 0.0,
    egger_line: bool = True,
    fill_contours: bool = True,
    xlabel: str = "Effect size",
    title: str = "Funnel Plot",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Funnel plot for publication-bias assessment in meta-analysis.

    Parameters
    ----------
    effects : sequence of float
        Study-level effect sizes.
    se : sequence of float
        Standard errors of the effect sizes.
    null_value : float
        Expected effect under the null hypothesis.
    egger_line : bool
        If True, overlay the Egger's regression test line.
    fill_contours : bool
        If True, shade pseudo-95% confidence contours.
    xlabel, title : str
        Axis label and title.
    theme : str or None
        MORIE theme name.
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    Sterne, J. A. C., & Egger, M. (2001). Funnel plots for detecting bias
    in meta-analysis. *Journal of Clinical Epidemiology*, 54(10), 1046-1055.
    https://doi.org/10.1016/S0895-4356(01)00377-8
    """
    effects = np.asarray(effects, dtype=float)
    se = np.asarray(se, dtype=float)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Contour shading (pseudo 95% CI)
        if fill_contours:
            se_range = np.linspace(0.001, se.max() * 1.15, 200)
            lower = null_value - 1.96 * se_range
            upper = null_value + 1.96 * se_range
            ax.fill_betweenx(se_range, lower, upper, color="#d0d0d0", alpha=0.4, label="95% pseudo-CI")

        ax.scatter(effects, se, s=40, color="#2c7bb6", edgecolors="white", linewidths=0.5, zorder=5)
        ax.axvline(null_value, color="grey", linestyle="--", linewidth=0.8)

        # Egger's test line
        if egger_line and len(effects) > 2:
            precision = 1.0 / se
            slope, intercept, _, p_val, _ = sp_stats.linregress(precision, effects * precision)
            se_fit = np.linspace(se.min() * 0.9, se.max() * 1.1, 100)
            pred_effects = intercept + slope / se_fit  # rearranged
            # Plot only over data range
            mask = (pred_effects > effects.min() - 1) & (pred_effects < effects.max() + 1)
            ax.plot(
                pred_effects[mask], se_fit[mask], color="#d7191c", linewidth=1.2, label=f"Egger line (p={p_val:.3f})"
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Standard Error")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Kaplan-Meier survival curve
# ---------------------------------------------------------------------------


def kaplan_meier_plot(
    time: np.ndarray | pd.Series,
    event: np.ndarray | pd.Series,
    *,
    group: np.ndarray | pd.Series | None = None,
    group_labels: dict[Any, str] | None = None,
    confidence: float = 0.95,
    show_ci: bool = True,
    show_censoring: bool = True,
    show_risk_table: bool = True,
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    title: str = "",
    colors: list[str] | None = None,
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Kaplan-Meier survival curves with confidence bands, censoring marks,
    and optional risk table.

    Parameters
    ----------
    time : array-like
        Observed times (follow-up duration or event time).
    event : array-like
        Event indicator (1 = event, 0 = censored).
    group : array-like or None
        Grouping variable for stratified curves.
    group_labels : dict or None
        Mapping from group values to display labels.
    confidence : float
        Confidence level for the bands (default 0.95).
    show_ci : bool
        Whether to display confidence bands.
    show_censoring : bool
        Whether to mark censored observations with ticks.
    show_risk_table : bool
        Whether to display the number-at-risk table below the curve.
    xlabel, ylabel, title : str
        Axis labels and title.
    colors : list of str or None
        Colours per group.  Uses default colour cycle if None.
    theme : str or None
        MORIE theme name.
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from
    incomplete observations. *Journal of the American Statistical
    Association*, 53(282), 457-481.
    https://doi.org/10.1080/01621459.1958.10501452
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)

    if group is None:
        groups_unique = [None]
        group = np.zeros(len(time), dtype=int)
    else:
        group = np.asarray(group)
        groups_unique = sorted(np.unique(group))

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(groups_unique))]

    with _apply_theme_ctx(theme):
        if show_risk_table:
            fig, (ax_main, ax_risk) = plt.subplots(
                2,
                1,
                figsize=(7, 5.5),
                gridspec_kw={"height_ratios": [4, 1]},
                sharex=True,
            )
        else:
            if ax is None:
                fig, ax_main = plt.subplots()
            else:
                ax_main = ax
                fig = ax.figure
            ax_risk = None

        risk_data: dict[str, list[tuple[float, int]]] = {}

        for idx, g in enumerate(groups_unique):
            mask = group == g if g is not None else np.ones(len(time), dtype=bool)
            t_g = time[mask]
            e_g = event[mask]

            # Sort by time
            order = np.argsort(t_g)
            t_g = t_g[order]
            e_g = e_g[order]

            unique_times = np.unique(t_g[e_g == 1])
            surv = 1.0
            surv_list = [(0.0, 1.0)]
            var_sum = 0.0
            ci_lo_list = [(0.0, 1.0)]
            ci_hi_list = [(0.0, 1.0)]
            n_risk = len(t_g)

            label = g
            if group_labels is not None and g in group_labels:
                label = group_labels[g]
            elif g is None:
                label = "Overall"

            risk_times_g: list[tuple[float, int]] = [(0.0, n_risk)]

            for ut in unique_times:
                # Number at risk just before this time
                n_risk = np.sum(t_g >= ut)
                d = np.sum((t_g == ut) & (e_g == 1))
                if n_risk > 0 and d > 0:
                    surv *= 1 - d / n_risk
                    var_sum += d / (n_risk * (n_risk - d)) if n_risk > d else 0
                surv_list.append((ut, surv))

                se = surv * np.sqrt(var_sum) if var_sum > 0 else 0
                ci_lo_list.append((ut, max(0, surv - z * se)))
                ci_hi_list.append((ut, min(1, surv + z * se)))
                risk_times_g.append((ut, n_risk - d))

            # Extend to max time
            t_max = t_g.max()
            surv_list.append((t_max, surv))
            ci_lo_list.append((t_max, ci_lo_list[-1][1]))
            ci_hi_list.append((t_max, ci_hi_list[-1][1]))

            step_t = [s[0] for s in surv_list]
            step_s = [s[1] for s in surv_list]

            ax_main.step(step_t, step_s, where="post", color=colors[idx], linewidth=1.5, label=str(label))

            if show_ci:
                lo_t = [s[0] for s in ci_lo_list]
                lo_s = [s[1] for s in ci_lo_list]
                hi_t = [s[0] for s in ci_hi_list]
                hi_s = [s[1] for s in ci_hi_list]
                ax_main.fill_between(lo_t, lo_s, hi_s, step="post", alpha=0.15, color=colors[idx])

            if show_censoring:
                cens_mask = e_g == 0
                cens_t = t_g[cens_mask]
                # Find survival at censoring times
                cens_s = []
                for ct in cens_t:
                    # Survival at this point
                    sv = 1.0
                    for st, ss in surv_list:
                        if st <= ct:
                            sv = ss
                        else:
                            break
                    cens_s.append(sv)
                ax_main.plot(cens_t, cens_s, "|", color=colors[idx], markersize=6, markeredgewidth=1)

            risk_data[str(label)] = risk_times_g

        ax_main.set_ylabel(ylabel)
        if title:
            ax_main.set_title(title)
        ax_main.set_ylim(-0.05, 1.05)
        ax_main.legend(frameon=False)

        # Risk table
        if ax_risk is not None:
            t_all = time
            tick_times = np.linspace(0, t_all.max(), min(6, int(t_all.max()) + 1))
            ax_risk.set_xlim(ax_main.get_xlim())
            for row_idx, (lbl, rt_list) in enumerate(risk_data.items()):
                rt_arr = np.array(rt_list)
                for tt in tick_times:
                    # Find n_risk at this time
                    mask_t = rt_arr[:, 0] <= tt
                    nr = int(rt_arr[mask_t, 1][-1]) if mask_t.any() else 0
                    ax_risk.text(
                        tt, row_idx, str(nr), ha="center", va="center", fontsize=8, color=colors[row_idx % len(colors)]
                    )
            ax_risk.set_yticks(range(len(risk_data)))
            ax_risk.set_yticklabels(list(risk_data.keys()), fontsize=8)
            ax_risk.set_xlabel(xlabel)
            ax_risk.set_title("Number at risk", fontsize=9, loc="left")
            ax_risk.tick_params(axis="x", which="both", length=0)
            ax_risk.spines["top"].set_visible(False)
            ax_risk.spines["right"].set_visible(False)
            ax_risk.spines["left"].set_visible(False)
            ax_risk.spines["bottom"].set_visible(False)
            ax_risk.grid(False)
        else:
            ax_main.set_xlabel(xlabel)

        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------


def roc_curve_plot(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    *,
    show_optimal: bool = True,
    show_ci: bool = True,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    label: str = "Model",
    title: str = "ROC Curve",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    ROC curve with AUC, optimal cutoff (Youden's J), and bootstrap CI.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities or decision scores.
    show_optimal : bool
        Mark the optimal threshold (max Youden's J).
    show_ci : bool
        Display bootstrap confidence band for the ROC.
    n_bootstrap : int
        Number of bootstrap replicates for the CI.
    confidence : float
        Confidence level.
    label : str
        Curve label in the legend.
    title : str
        Plot title.
    theme : str or None
        MORIE theme name.
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve as sk_roc_curve

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    fpr, tpr, thresholds = sk_roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.plot(fpr, tpr, color="#2c7bb6", linewidth=1.8, label=f"{label} (AUC = {auc_val:.3f})")

        # Bootstrap CI for ROC
        if show_ci and len(y_true) > 20:
            rng = np.random.default_rng(42)
            boot_tprs = []
            mean_fpr = np.linspace(0, 1, 100)
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(y_true), size=len(y_true))
                if len(np.unique(y_true[idx])) < 2:
                    continue
                fpr_b, tpr_b, _ = sk_roc_curve(y_true[idx], y_score[idx])
                boot_tprs.append(np.interp(mean_fpr, fpr_b, tpr_b))
            if boot_tprs:
                boot_tprs = np.array(boot_tprs)
                alpha = (1 - confidence) / 2
                lo = np.percentile(boot_tprs, 100 * alpha, axis=0)
                hi = np.percentile(boot_tprs, 100 * (1 - alpha), axis=0)
                ax.fill_between(mean_fpr, lo, hi, alpha=0.15, color="#2c7bb6")

        # Optimal cutoff (Youden's J)
        if show_optimal:
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            ax.plot(
                fpr[best_idx],
                tpr[best_idx],
                "o",
                color="#d7191c",
                markersize=8,
                label=f"Optimal (t={thresholds[best_idx]:.3f})",
            )

        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Precision-recall curve
# ---------------------------------------------------------------------------


def precision_recall_plot(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    *,
    label: str = "Model",
    title: str = "Precision-Recall Curve",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Precision-recall curve with average precision (AP).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities.
    label, title : str
        Legend label and title.
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve as sk_pr_curve

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    prec, rec, _ = sk_pr_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    prevalence = y_true.mean()

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.step(rec, prec, where="post", color="#2c7bb6", linewidth=1.8, label=f"{label} (AP = {ap:.3f})")
        ax.axhline(prevalence, color="grey", linestyle="--", linewidth=0.8, label=f"Baseline ({prevalence:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(frameon=False)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------


def calibration_plot(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
    show_histogram: bool = True,
    title: str = "Calibration Plot",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Calibration plot (observed vs predicted probability).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities.
    n_bins : int
        Number of bins.
    strategy : str
        ``"quantile"`` or ``"uniform"``.
    show_histogram : bool
        Show a histogram of predicted probabilities beneath the calibration
        curve.
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    Van Calster, B., et al. (2019). Calibration: the Achilles heel of
    predictive analytics. *BMC Medicine*, 17(1), 230.
    https://doi.org/10.1186/s12916-019-1466-7
    """
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=strategy)

    with _apply_theme_ctx(theme):
        if show_histogram:
            fig, (ax_cal, ax_hist) = plt.subplots(
                2,
                1,
                figsize=(5.5, 5.5),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )
        else:
            if ax is None:
                fig, ax_cal = plt.subplots()
            else:
                ax_cal = ax
                fig = ax.figure
            ax_hist = None

        ax_cal.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8, label="Perfectly calibrated")
        ax_cal.plot(prob_pred, prob_true, "s-", color="#2c7bb6", linewidth=1.5, markersize=6, label="Model")
        ax_cal.set_ylabel("Observed Probability")
        ax_cal.set_title(title)
        ax_cal.legend(frameon=False)
        ax_cal.set_xlim(-0.02, 1.02)
        ax_cal.set_ylim(-0.02, 1.02)

        if ax_hist is not None:
            ax_hist.hist(y_pred, bins=50, color="#abd9e9", edgecolor="white", linewidth=0.3)
            ax_hist.set_xlabel("Predicted Probability")
            ax_hist.set_ylabel("Count")

        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Balance / Love plot
# ---------------------------------------------------------------------------


def balance_plot(
    smd_before: pd.Series | dict[str, float],
    smd_after: pd.Series | dict[str, float] | None = None,
    *,
    threshold: float = 0.1,
    title: str = "Covariate Balance",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Love plot showing standardised mean differences (SMD) before and after
    matching/weighting.

    Parameters
    ----------
    smd_before : dict or Series
        Absolute SMD for each covariate before adjustment.
    smd_after : dict, Series, or None
        Absolute SMD after adjustment.  If ``None``, only pre-adjustment
        values are shown.
    threshold : float
        SMD threshold line (commonly 0.1).
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    Austin, P. C. (2009). Balance diagnostics for comparing the distribution
    of baseline covariates between treatment groups in propensity-score
    matched samples. *Statistics in Medicine*, 28(25), 3083-3107.
    https://doi.org/10.1002/sim.3697
    """
    if isinstance(smd_before, dict):
        smd_before = pd.Series(smd_before)
    labels = list(smd_before.index)
    n = len(labels)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * n + 1)))
        else:
            fig = ax.figure

        y = np.arange(n)
        ax.scatter(
            np.abs(smd_before.values),
            y,
            marker="o",
            s=60,
            color="#d7191c",
            label="Before",
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

        if smd_after is not None:
            if isinstance(smd_after, dict):
                smd_after = pd.Series(smd_after)
            ax.scatter(
                np.abs(smd_after.reindex(labels).values),
                y,
                marker="s",
                s=60,
                color="#2c7bb6",
                label="After",
                zorder=5,
                edgecolors="white",
                linewidths=0.5,
            )
            # Lines connecting before/after
            for i in range(n):
                lbl = labels[i]
                if lbl in smd_after.index:
                    ax.plot(
                        [np.abs(smd_before[lbl]), np.abs(smd_after[lbl])], [i, i], color="grey", linewidth=0.6, zorder=2
                    )

        ax.axvline(threshold, color="grey", linestyle="--", linewidth=0.8, label=f"Threshold ({threshold})")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("|Standardised Mean Difference|")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# DAG (directed acyclic graph) -- pure matplotlib, no graphviz
# ---------------------------------------------------------------------------


def dag_plot(
    edges: list[tuple[str, str]],
    *,
    node_positions: dict[str, tuple[float, float]] | None = None,
    highlight_path: list[str] | None = None,
    title: str = "",
    node_color: str = "#2c7bb6",
    highlight_color: str = "#d7191c",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Directed acyclic graph visualisation using matplotlib (no graphviz).

    Parameters
    ----------
    edges : list of (str, str)
        Directed edges as ``(source, target)`` tuples.
    node_positions : dict or None
        ``{node_name: (x, y)}`` positions.  If ``None``, a simple
        topological layout is computed.
    highlight_path : list of str or None
        A causal path to highlight in a different colour.
    title : str
    node_color, highlight_color : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    nodes: list[str] = []
    for a, b in edges:
        if a not in nodes:
            nodes.append(a)
        if b not in nodes:
            nodes.append(b)

    # Build adjacency for topological sort
    children: dict[str, list[str]] = {n: [] for n in nodes}
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    for a, b in edges:
        children[a].append(b)
        in_degree[b] += 1

    # Simple layered layout
    if node_positions is None:

        layers: list[list[str]] = []
        remaining = set(nodes)
        deg = dict(in_degree)
        while remaining:
            layer = [n for n in remaining if deg.get(n, 0) == 0]
            if not layer:
                layer = [remaining.pop()]
            layers.append(layer)
            for n in layer:
                remaining.discard(n)
                for c in children.get(n, []):
                    deg[c] = deg.get(c, 0) - 1
        node_positions = {}
        for li, layer in enumerate(layers):
            for ni, n in enumerate(layer):
                x = li * 2.0
                y = -(ni - (len(layer) - 1) / 2.0) * 1.5
                node_positions[n] = (x, y)

    highlight_edges = set()
    if highlight_path is not None:
        for i in range(len(highlight_path) - 1):
            highlight_edges.add((highlight_path[i], highlight_path[i + 1]))

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, len(nodes) * 1.2), max(4, len(nodes) * 0.8)))
        else:
            fig = ax.figure

        # Draw edges (arrows)
        for a, b in edges:
            x0, y0 = node_positions[a]
            x1, y1 = node_positions[b]
            ec = highlight_color if (a, b) in highlight_edges else "#888888"
            lw = 2.0 if (a, b) in highlight_edges else 1.0
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ec,
                    lw=lw,
                    connectionstyle="arc3,rad=0.1",
                    shrinkA=18,
                    shrinkB=18,
                ),
            )

        # Draw nodes
        for n, (x, y) in node_positions.items():
            nc = highlight_color if (highlight_path and n in highlight_path) else node_color
            circle = mpatches.FancyBboxPatch(
                (x - 0.55, y - 0.35),
                1.1,
                0.7,
                boxstyle="round,pad=0.1",
                facecolor=nc,
                edgecolor="white",
                linewidth=2,
                alpha=0.85,
            )
            ax.add_patch(circle)
            ax.text(x, y, n, ha="center", va="center", fontsize=9, fontweight="bold", color="white")

        ax.set_aspect("equal")
        ax.autoscale()
        ax.margins(0.15)
        ax.axis("off")
        if title:
            ax.set_title(title)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# QQ plot
# ---------------------------------------------------------------------------


def qq_plot(
    data: np.ndarray | pd.Series,
    *,
    dist: str = "norm",
    dist_params: tuple = (),
    title: str = "Q-Q Plot",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Quantile-quantile plot against a theoretical distribution.

    Parameters
    ----------
    data : array-like
        Observed data.
    dist : str
        Name of a ``scipy.stats`` distribution (default ``"norm"``).
    dist_params : tuple
        Additional shape parameters for the distribution (loc and scale are
        estimated from *data*).
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        res = sp_stats.probplot(data, dist=dist, sparams=dist_params, plot=None)
        theoretical, ordered = res[0]
        slope, intercept, _ = res[1]

        ax.scatter(theoretical, ordered, s=20, color="#2c7bb6", edgecolors="white", linewidths=0.3, zorder=5)
        x_line = np.array([theoretical.min(), theoretical.max()])
        ax.plot(x_line, slope * x_line + intercept, "--", color="#d7191c", linewidth=1)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title(title)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Residual diagnostics (4-panel)
# ---------------------------------------------------------------------------


def residual_diagnostic_plots(
    fitted: np.ndarray | pd.Series,
    residuals: np.ndarray | pd.Series,
    *,
    leverage: np.ndarray | pd.Series | None = None,
    cooks_d: np.ndarray | pd.Series | None = None,
    title: str = "Residual Diagnostics",
    theme: str | None = "publication",
) -> plt.Figure:
    """
    Four-panel residual diagnostic plots: residuals vs fitted,
    scale-location, Q-Q of residuals, and residuals vs leverage.

    Parameters
    ----------
    fitted : array-like
        Fitted (predicted) values.
    residuals : array-like
        Model residuals.
    leverage : array-like or None
        Hat-matrix diagonal values (leverage).  If ``None``, the leverage
        panel shows a histogram of residuals instead.
    cooks_d : array-like or None
        Cook's distance.  If provided, points exceeding 4/n are highlighted.
    title : str
    theme : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    fitted = np.asarray(fitted, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    std_resid = residuals / (np.std(residuals) + 1e-12)

    with _apply_theme_ctx(theme):
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        fig.suptitle(title, fontsize=12)

        # 1. Residuals vs Fitted
        ax = axes[0, 0]
        ax.scatter(fitted, residuals, s=15, alpha=0.6, color="#2c7bb6", edgecolors="none")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        # LOWESS smoother
        try:
            import statsmodels.api as sm_api

            lowess = sm_api.nonparametric.lowess(residuals, fitted, frac=0.6)
            ax.plot(lowess[:, 0], lowess[:, 1], color="#d7191c", linewidth=1.2)
        except Exception:
            pass
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")

        # 2. Q-Q of residuals
        ax = axes[0, 1]
        res_qq = sp_stats.probplot(std_resid, plot=None)
        ax.scatter(res_qq[0][0], res_qq[0][1], s=15, alpha=0.6, color="#2c7bb6", edgecolors="none")
        x_line = np.array([res_qq[0][0].min(), res_qq[0][0].max()])
        ax.plot(x_line, res_qq[1][0] * x_line + res_qq[1][1], "--", color="#d7191c", linewidth=1)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Standardised Residuals")
        ax.set_title("Normal Q-Q")

        # 3. Scale-Location
        ax = axes[1, 0]
        sqrt_abs_resid = np.sqrt(np.abs(std_resid))
        ax.scatter(fitted, sqrt_abs_resid, s=15, alpha=0.6, color="#2c7bb6", edgecolors="none")
        try:
            lowess2 = sm_api.nonparametric.lowess(sqrt_abs_resid, fitted, frac=0.6)
            ax.plot(lowess2[:, 0], lowess2[:, 1], color="#d7191c", linewidth=1.2)
        except Exception:
            pass
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("sqrt(|Standardised Residuals|)")
        ax.set_title("Scale-Location")

        # 4. Residuals vs Leverage or Cook's distance
        ax = axes[1, 1]
        if leverage is not None:
            leverage = np.asarray(leverage, dtype=float)
            ax.scatter(leverage, std_resid, s=15, alpha=0.6, color="#2c7bb6", edgecolors="none")
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            if cooks_d is not None:
                cooks_d = np.asarray(cooks_d, dtype=float)
                threshold = 4.0 / len(fitted)
                influential = cooks_d > threshold
                ax.scatter(
                    leverage[influential],
                    std_resid[influential],
                    s=40,
                    facecolors="none",
                    edgecolors="#d7191c",
                    linewidths=1.2,
                )
            ax.set_xlabel("Leverage")
            ax.set_ylabel("Standardised Residuals")
            ax.set_title("Residuals vs Leverage")
        else:
            ax.hist(residuals, bins=30, color="#abd9e9", edgecolor="white", linewidth=0.3)
            ax.set_xlabel("Residuals")
            ax.set_ylabel("Frequency")
            ax.set_title("Residual Distribution")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Influence plot
# ---------------------------------------------------------------------------


def influence_plot(
    leverage: np.ndarray | pd.Series,
    cooks_d: np.ndarray | pd.Series,
    residuals: np.ndarray | pd.Series,
    *,
    labels: Sequence[str] | None = None,
    title: str = "Influence Plot",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Influence plot: Cook's distance vs leverage, sized by residuals.

    Parameters
    ----------
    leverage : array-like
        Hat-matrix diagonal values.
    cooks_d : array-like
        Cook's distance.
    residuals : array-like
        Model residuals (used to size markers).
    labels : sequence of str or None
        Observation labels for the most influential points.
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    leverage = np.asarray(leverage, dtype=float)
    cooks_d = np.asarray(cooks_d, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    n = len(leverage)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        sizes = 20 + 200 * (np.abs(residuals) / (np.abs(residuals).max() + 1e-12))
        ax.scatter(leverage, cooks_d, s=sizes, alpha=0.6, color="#2c7bb6", edgecolors="white", linewidths=0.5)

        threshold = 4.0 / n
        ax.axhline(threshold, color="#d7191c", linestyle="--", linewidth=0.8, label=f"Cook's D = {threshold:.4f}")

        # Label top influential
        if labels is not None:
            top_idx = np.argsort(cooks_d)[-3:]
            for i in top_idx:
                if cooks_d[i] > threshold:
                    ax.annotate(str(labels[i]), (leverage[i], cooks_d[i]), fontsize=7, ha="left", va="bottom")

        ax.set_xlabel("Leverage")
        ax.set_ylabel("Cook's Distance")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Added variable plot (partial regression)
# ---------------------------------------------------------------------------


def added_variable_plot(
    data: pd.DataFrame,
    outcome: str,
    focus_var: str,
    covariates: list[str],
    *,
    title: str | None = None,
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Added variable plot (partial regression plot) for a single covariate.

    Residualises both *outcome* and *focus_var* on the remaining covariates
    and scatterplots the residuals.

    Parameters
    ----------
    data : DataFrame
        Input data.
    outcome : str
        Outcome variable name.
    focus_var : str
        The covariate to examine.
    covariates : list of str
        All covariates including *focus_var*.
    title : str or None
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.linear_model import LinearRegression

    other = [c for c in covariates if c != focus_var]
    complete = data[[outcome, focus_var] + other].dropna()
    if len(other) == 0:
        raise ValueError("Need at least one other covariate besides focus_var")

    X_other = complete[other].values
    # Residualise outcome
    m_y = LinearRegression().fit(X_other, complete[outcome].values)
    resid_y = complete[outcome].values - m_y.predict(X_other)
    # Residualise focus_var
    m_x = LinearRegression().fit(X_other, complete[focus_var].values)
    resid_x = complete[focus_var].values - m_x.predict(X_other)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.scatter(resid_x, resid_y, s=15, alpha=0.5, color="#2c7bb6", edgecolors="none")
        # Fit line
        slope, intercept, r, p, se = sp_stats.linregress(resid_x, resid_y)
        x_line = np.linspace(resid_x.min(), resid_x.max(), 100)
        ax.plot(
            x_line, slope * x_line + intercept, color="#d7191c", linewidth=1.2, label=f"slope={slope:.3f}, p={p:.3g}"
        )
        ax.set_xlabel(f"{focus_var} | others")
        ax.set_ylabel(f"{outcome} | others")
        ax.set_title(title or f"Added Variable Plot: {focus_var}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Component-plus-residual plot
# ---------------------------------------------------------------------------


def component_residual_plot(
    data: pd.DataFrame,
    outcome: str,
    focus_var: str,
    covariates: list[str],
    *,
    title: str | None = None,
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Component-plus-residual (partial residual) plot.

    Plots ``beta_j * x_j + residuals`` against ``x_j`` to detect
    non-linearity in the effect of *focus_var*.

    Parameters
    ----------
    data : DataFrame
    outcome : str
    focus_var : str
    covariates : list of str
    title : str or None
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import statsmodels.api as sm_api

    complete = data[[outcome] + covariates].dropna()
    X = sm_api.add_constant(complete[covariates])
    model = sm_api.OLS(complete[outcome], X).fit()
    beta_j = model.params.get(focus_var, 0.0)
    partial_resid = model.resid + beta_j * complete[focus_var]

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.scatter(complete[focus_var], partial_resid, s=15, alpha=0.5, color="#2c7bb6", edgecolors="none")
        # Component line
        x_sorted = np.sort(complete[focus_var].values)
        ax.plot(x_sorted, beta_j * x_sorted, color="#d7191c", linewidth=1.2, label=f"Component (beta={beta_j:.3f})")
        # LOWESS
        try:
            lowess = sm_api.nonparametric.lowess(partial_resid.values, complete[focus_var].values, frac=0.6)
            ax.plot(lowess[:, 0], lowess[:, 1], color="#fdae61", linewidth=1.2, label="LOWESS")
        except Exception:
            pass
        ax.set_xlabel(focus_var)
        ax.set_ylabel(f"Component + Residual ({outcome})")
        ax.set_title(title or f"Component+Residual: {focus_var}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bland-Altman plot
# ---------------------------------------------------------------------------


def bland_altman_plot(
    method1: np.ndarray | pd.Series,
    method2: np.ndarray | pd.Series,
    *,
    confidence: float = 0.95,
    title: str = "Bland-Altman Plot",
    xlabel: str = "Mean of Methods",
    ylabel: str = "Difference (Method 1 - Method 2)",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Bland-Altman plot for method agreement assessment.

    Parameters
    ----------
    method1, method2 : array-like
        Measurements from two methods on the same subjects.
    confidence : float
        Confidence level for limits of agreement (default 0.95).
    title, xlabel, ylabel : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    Bland, J. M., & Altman, D. G. (1986). Statistical methods for
    assessing agreement between two methods of clinical measurement.
    *The Lancet*, 327(8476), 307-310.
    https://doi.org/10.1016/S0140-6736(86)90837-8
    """
    m1 = np.asarray(method1, dtype=float)
    m2 = np.asarray(method2, dtype=float)
    mean_vals = (m1 + m2) / 2
    diff_vals = m1 - m2
    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals, ddof=1)
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.scatter(mean_vals, diff_vals, s=25, alpha=0.6, color="#2c7bb6", edgecolors="white", linewidths=0.3)
        ax.axhline(mean_diff, color="#2c7bb6", linewidth=1, label=f"Mean diff = {mean_diff:.3f}")
        ax.axhline(
            mean_diff + z * sd_diff,
            color="#d7191c",
            linestyle="--",
            linewidth=0.8,
            label=f"+{z:.2f} SD = {mean_diff + z * sd_diff:.3f}",
        )
        ax.axhline(
            mean_diff - z * sd_diff,
            color="#d7191c",
            linestyle="--",
            linewidth=0.8,
            label=f"-{z:.2f} SD = {mean_diff - z * sd_diff:.3f}",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------


def correlation_heatmap(
    data: pd.DataFrame,
    *,
    method: str = "pearson",
    show_stars: bool = True,
    mask_upper: bool = True,
    cmap: str = "RdBu_r",
    title: str = "Correlation Matrix",
    fmt: str = ".2f",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Correlation heatmap with optional significance stars.

    Parameters
    ----------
    data : DataFrame
        Numeric columns to correlate.
    method : str
        ``"pearson"``, ``"spearman"``, or ``"kendall"``.
    show_stars : bool
        Annotate cells with significance stars (* p<0.05, ** p<0.01,
        *** p<0.001).
    mask_upper : bool
        Mask the upper triangle.
    cmap : str
        Matplotlib colourmap.
    title : str
    fmt : str
        Number format for annotation.
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    numeric = data.select_dtypes(include="number")
    corr = numeric.corr(method=method)
    n_vars = len(corr)

    # Compute p-values
    pval_mat = np.ones((n_vars, n_vars))
    cols = corr.columns
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            valid = numeric[[cols[i], cols[j]]].dropna()
            if len(valid) > 2:
                if method == "pearson":
                    _, p = sp_stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                elif method == "spearman":
                    _, p = sp_stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
                else:
                    _, p = sp_stats.kendalltau(valid.iloc[:, 0], valid.iloc[:, 1])
                pval_mat[i, j] = p
                pval_mat[j, i] = p

    with _apply_theme_ctx(theme):
        if ax is None:
            sz = max(5, n_vars * 0.6)
            fig, ax = plt.subplots(figsize=(sz, sz))
        else:
            fig = ax.figure

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1) if mask_upper else None

        vals = corr.values.copy()
        if mask is not None:
            display = np.where(~mask, vals, np.nan)
        else:
            display = vals

        im = ax.imshow(display, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate
        for i in range(n_vars):
            for j in range(n_vars):
                if mask is not None and mask[i, j]:
                    continue
                val = corr.values[i, j]
                stars = ""
                if show_stars and i != j:
                    p = pval_mat[i, j]
                    if p < 0.001:
                        stars = "***"
                    elif p < 0.01:
                        stars = "**"
                    elif p < 0.05:
                        stars = "*"
                text = f"{val:{fmt}}{stars}"
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=text_color)

        ax.set_xticks(range(n_vars))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(cols)
        ax.set_title(title)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Missing data pattern heatmap
# ---------------------------------------------------------------------------


def missing_pattern_plot(
    data: pd.DataFrame,
    *,
    max_cols: int = 50,
    title: str = "Missing Data Pattern",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Heatmap of missing data patterns (rows = observations, cols = variables).

    Parameters
    ----------
    data : DataFrame
    max_cols : int
        Maximum number of columns to display (sorted by missingness).
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    miss = data.isnull()
    miss_pct = miss.mean().sort_values(ascending=False)
    cols_show = miss_pct.index[:max_cols]
    miss_subset = miss[cols_show]

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, len(cols_show) * 0.3), 5))
        else:
            fig = ax.figure

        ax.imshow(miss_subset.values.astype(float), aspect="auto", cmap="Greys", interpolation="none")
        ax.set_xticks(range(len(cols_show)))
        ax.set_xticklabels(cols_show, rotation=90, fontsize=6)
        ax.set_ylabel("Observation")
        ax.set_title(title)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Distribution comparison plots
# ---------------------------------------------------------------------------


def distribution_comparison_plot(
    groups: dict[str, np.ndarray | pd.Series],
    *,
    kind: str = "violin",
    title: str = "",
    xlabel: str = "Value",
    colors: list[str] | None = None,
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Compare distributions across groups using violin, histogram, or
    ridge plots.

    Parameters
    ----------
    groups : dict
        ``{group_label: values}`` mapping.
    kind : str
        ``"violin"``, ``"histogram"``, or ``"ridge"``.
    title : str
    xlabel : str
    colors : list of str or None
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(groups.keys())
    data_list = [np.asarray(groups[k], dtype=float) for k in labels]
    n_groups = len(labels)

    if colors is None:
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(n_groups)]

    with _apply_theme_ctx(theme):
        if kind == "violin":
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            parts = ax.violinplot(data_list, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts.get("bodies", [])):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
            ax.set_xticks(range(1, n_groups + 1))
            ax.set_xticklabels(labels)
            ax.set_ylabel(xlabel)
            ax.set_title(title)
            fig.tight_layout()

        elif kind == "histogram":
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            for i, (lbl, vals) in enumerate(zip(labels, data_list)):
                ax.hist(vals, bins=30, alpha=0.5, color=colors[i], label=lbl, edgecolor="white", linewidth=0.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frequency")
            ax.set_title(title)
            ax.legend(frameon=False)
            fig.tight_layout()

        elif kind == "ridge":
            fig, axes_list = plt.subplots(n_groups, 1, figsize=(6, 1.5 * n_groups), sharex=True)
            if n_groups == 1:
                axes_list = [axes_list]
            for i, (ax_r, lbl, vals) in enumerate(zip(axes_list, labels, data_list)):
                ax_r.fill_between(
                    *_kde_line(vals),
                    alpha=0.6,
                    color=colors[i],
                )
                ax_r.set_ylabel(lbl, rotation=0, ha="right", va="center", fontsize=9)
                ax_r.set_yticks([])
                for spine in ["top", "right", "left"]:
                    ax_r.spines[spine].set_visible(False)
            axes_list[-1].set_xlabel(xlabel)
            fig.suptitle(title, fontsize=11)
            fig.tight_layout()
        else:
            raise ValueError(f"Unknown kind '{kind}'. Use 'violin', 'histogram', or 'ridge'.")
    return fig


def _kde_line(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute KDE for a ridge plot."""
    from scipy.stats import gaussian_kde

    vals = values[np.isfinite(values)]
    if len(vals) < 2:
        return np.array([0]), np.array([0])
    kde = gaussian_kde(vals)
    x = np.linspace(vals.min() - 0.5 * vals.std(), vals.max() + 0.5 * vals.std(), 200)
    return x, kde(x)


# ---------------------------------------------------------------------------
# Effect size comparison plot
# ---------------------------------------------------------------------------


def effect_size_plot(
    labels: Sequence[str],
    effects: Sequence[float],
    ci_lower: Sequence[float],
    ci_upper: Sequence[float],
    *,
    colors: list[str] | None = None,
    title: str = "Effect Size Comparison",
    xlabel: str = "Effect Size",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of effect sizes with confidence intervals.

    Parameters
    ----------
    labels : sequence of str
    effects : sequence of float
    ci_lower, ci_upper : sequence of float
    colors : list of str or None
    title, xlabel : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(labels)
    effects = np.asarray(effects, dtype=float)
    ci_lower = np.asarray(ci_lower, dtype=float)
    ci_upper = np.asarray(ci_upper, dtype=float)
    err_lo = effects - ci_lower
    err_hi = ci_upper - effects

    if colors is None:
        colors = ["#2c7bb6"] * n

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * n)))
        else:
            fig = ax.figure

        y = np.arange(n)
        ax.barh(y, effects, xerr=[err_lo, err_hi], color=colors, height=0.6, capsize=3, ecolor="grey")
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Power curve
# ---------------------------------------------------------------------------


def power_curve_plot(
    sample_sizes: Sequence[int],
    power_values: Sequence[float] | None = None,
    *,
    effect_size: float = 0.5,
    alpha: float = 0.05,
    test: str = "two_sample_t",
    target_power: float = 0.80,
    title: str = "Statistical Power Curve",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Power curve showing statistical power as a function of sample size.

    Parameters
    ----------
    sample_sizes : sequence of int
        Sample sizes to evaluate.
    power_values : sequence of float or None
        Pre-computed power values.  If ``None``, computed for a two-sample
        t-test using a normal approximation.
    effect_size : float
        Cohen's d or equivalent (used only when *power_values* is None).
    alpha : float
        Significance level.
    test : str
        Currently ``"two_sample_t"`` is supported for auto-computation.
    target_power : float
        Target power level (horizontal line).
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    ns = np.asarray(sample_sizes, dtype=float)

    if power_values is None:
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        pwr = []
        for n in ns:
            se = np.sqrt(2 / n)
            z_beta = effect_size / se - z_alpha
            pwr.append(sp_stats.norm.cdf(z_beta))
        power_values = np.array(pwr)
    else:
        power_values = np.asarray(power_values, dtype=float)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.plot(ns, power_values, "-o", color="#2c7bb6", linewidth=1.8, markersize=4)
        ax.axhline(target_power, color="#d7191c", linestyle="--", linewidth=0.8, label=f"Target = {target_power}")
        ax.set_xlabel("Sample Size (per group)")
        ax.set_ylabel("Power")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(frameon=False)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Propensity score distribution
# ---------------------------------------------------------------------------


def propensity_score_plot(
    ps: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    *,
    title: str = "Propensity Score Distribution",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Overlaid histograms of propensity scores for treated and control groups.

    Parameters
    ----------
    ps : array-like
        Propensity scores.
    treatment : array-like
        Binary treatment indicator.
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    ps = np.asarray(ps, dtype=float)
    treatment = np.asarray(treatment, dtype=int)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.hist(
            ps[treatment == 1],
            bins=40,
            alpha=0.55,
            color="#d7191c",
            label="Treated",
            density=True,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.hist(
            ps[treatment == 0],
            bins=40,
            alpha=0.55,
            color="#2c7bb6",
            label="Control",
            density=True,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CATE heterogeneity plot
# ---------------------------------------------------------------------------


def cate_heterogeneity_plot(
    cate_values: np.ndarray | pd.Series,
    *,
    covariate: np.ndarray | pd.Series | None = None,
    covariate_name: str = "Covariate",
    title: str = "CATE Heterogeneity",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot conditional average treatment effect heterogeneity.

    If *covariate* is provided, CATE is plotted against it; otherwise a
    histogram of CATE values is shown.

    Parameters
    ----------
    cate_values : array-like
        Individual-level CATE estimates.
    covariate : array-like or None
        A continuous covariate to plot CATE against.
    covariate_name : str
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    cate = np.asarray(cate_values, dtype=float)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if covariate is not None:
            cov = np.asarray(covariate, dtype=float)
            ax.scatter(cov, cate, s=12, alpha=0.4, color="#2c7bb6", edgecolors="none")
            # LOWESS
            try:
                import statsmodels.api as sm_api

                lowess = sm_api.nonparametric.lowess(cate, cov, frac=0.5)
                ax.plot(lowess[:, 0], lowess[:, 1], color="#d7191c", linewidth=1.5)
            except Exception:
                pass
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            ax.set_xlabel(covariate_name)
            ax.set_ylabel("CATE")
        else:
            ax.hist(cate, bins=40, color="#abd9e9", edgecolor="white", linewidth=0.3)
            ax.axvline(np.mean(cate), color="#d7191c", linewidth=1.2, label=f"Mean = {np.mean(cate):.3f}")
            ax.set_xlabel("CATE")
            ax.set_ylabel("Frequency")
            ax.legend(frameon=False)

        ax.set_title(title)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Event study plot
# ---------------------------------------------------------------------------


def event_study_plot(
    periods: Sequence[int],
    estimates: Sequence[float],
    ci_lower: Sequence[float],
    ci_upper: Sequence[float],
    *,
    treatment_period: int = 0,
    title: str = "Event Study",
    xlabel: str = "Periods Relative to Treatment",
    ylabel: str = "Estimate",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Event study plot with pre/post periods and confidence bands.

    Parameters
    ----------
    periods : sequence of int
        Relative time periods (negative = pre, positive = post).
    estimates : sequence of float
        Point estimates for each period.
    ci_lower, ci_upper : sequence of float
        Confidence interval bounds.
    treatment_period : int
        The period at which treatment begins (vertical line).
    title, xlabel, ylabel : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    periods = np.asarray(periods, dtype=int)
    estimates = np.asarray(estimates, dtype=float)
    ci_lower = np.asarray(ci_lower, dtype=float)
    ci_upper = np.asarray(ci_upper, dtype=float)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Pre / post colouring
        pre_mask = periods < treatment_period
        post_mask = periods >= treatment_period

        ax.fill_between(periods, ci_lower, ci_upper, alpha=0.15, color="#2c7bb6")
        ax.plot(
            periods[pre_mask],
            estimates[pre_mask],
            "o-",
            color="#888888",
            linewidth=1.5,
            markersize=5,
            label="Pre-treatment",
        )
        ax.plot(
            periods[post_mask],
            estimates[post_mask],
            "o-",
            color="#2c7bb6",
            linewidth=1.5,
            markersize=5,
            label="Post-treatment",
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(treatment_period - 0.5, color="#d7191c", linestyle=":", linewidth=1, label="Treatment onset")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Waterfall plot
# ---------------------------------------------------------------------------


def waterfall_plot(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "Waterfall Plot",
    ylabel: str = "Value",
    colors: tuple[str, str] = ("#2c7bb6", "#d7191c"),
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Waterfall plot showing sequential contributions to a total.

    Parameters
    ----------
    labels : sequence of str
        Step labels.
    values : sequence of float
        Incremental changes at each step.
    title, ylabel : str
    colors : tuple of (positive_color, negative_color)
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(values)
    bottoms = np.concatenate([[0], cumulative[:-1]])

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.7), 4))
        else:
            fig = ax.figure

        bar_colors = [colors[0] if v >= 0 else colors[1] for v in values]
        ax.bar(
            range(len(labels)), values, bottom=bottoms, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.7
        )

        # Connection lines
        for i in range(len(labels) - 1):
            ax.plot([i + 0.35, i + 0.65], [cumulative[i], cumulative[i]], color="grey", linewidth=0.6)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="grey", linewidth=0.5)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Volcano plot
# ---------------------------------------------------------------------------


def volcano_plot(
    log2_fc: np.ndarray | pd.Series,
    neg_log10_p: np.ndarray | pd.Series,
    *,
    labels: Sequence[str] | None = None,
    fc_threshold: float = 1.0,
    p_threshold: float = 1.301,  # -log10(0.05)
    title: str = "Volcano Plot",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Volcano plot (effect size vs statistical significance).

    Parameters
    ----------
    log2_fc : array-like
        Log2 fold change (or any effect-size metric).
    neg_log10_p : array-like
        -log10(p-value).
    labels : sequence of str or None
        Feature labels for significant points.
    fc_threshold : float
        Fold-change threshold for highlighting.
    p_threshold : float
        -log10(p) threshold (default 1.301 = -log10(0.05)).
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    fc = np.asarray(log2_fc, dtype=float)
    pv = np.asarray(neg_log10_p, dtype=float)

    sig = (np.abs(fc) > fc_threshold) & (pv > p_threshold)
    not_sig = ~sig

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.scatter(fc[not_sig], pv[not_sig], s=12, alpha=0.4, color="#888888", edgecolors="none")
        ax.scatter(fc[sig], pv[sig], s=20, alpha=0.7, color="#d7191c", edgecolors="none")
        ax.axhline(p_threshold, color="grey", linestyle="--", linewidth=0.7)
        ax.axvline(fc_threshold, color="grey", linestyle="--", linewidth=0.7)
        ax.axvline(-fc_threshold, color="grey", linestyle="--", linewidth=0.7)

        if labels is not None:
            top_idx = np.where(sig)[0]
            sorted_top = top_idx[np.argsort(pv[top_idx])[::-1]][:10]
            for i in sorted_top:
                ax.annotate(str(labels[i]), (fc[i], pv[i]), fontsize=6, ha="center")

        ax.set_xlabel("log2(Fold Change)")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(title)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Manhattan plot
# ---------------------------------------------------------------------------


def manhattan_plot(
    positions: np.ndarray | pd.Series,
    neg_log10_p: np.ndarray | pd.Series,
    *,
    chromosomes: np.ndarray | pd.Series | None = None,
    significance: float = 5e-8,
    suggestive: float = 1e-5,
    title: str = "Manhattan Plot",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Manhattan plot for genome-wide or multiple-testing results.

    Parameters
    ----------
    positions : array-like
        Genomic position or test index.
    neg_log10_p : array-like
        -log10(p-value).
    chromosomes : array-like or None
        Chromosome/group assignment for alternating colours.
    significance : float
        Genome-wide significance threshold (p-value scale).
    suggestive : float
        Suggestive significance threshold (p-value scale).
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    pos = np.asarray(positions, dtype=float)
    pv = np.asarray(neg_log10_p, dtype=float)

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure

        if chromosomes is not None:
            chrom = np.asarray(chromosomes)
            unique_chrom = np.unique(chrom)
            colors_cycle = ["#2c7bb6", "#abd9e9"]
            for i, c in enumerate(unique_chrom):
                mask = chrom == c
                ax.scatter(pos[mask], pv[mask], s=8, alpha=0.6, color=colors_cycle[i % 2], edgecolors="none")
        else:
            ax.scatter(pos, pv, s=8, alpha=0.6, color="#2c7bb6", edgecolors="none")

        ax.axhline(
            -np.log10(significance), color="#d7191c", linestyle="--", linewidth=0.8, label=f"p = {significance:.0e}"
        )
        ax.axhline(-np.log10(suggestive), color="#fdae61", linestyle="--", linewidth=0.8, label=f"p = {suggestive:.0e}")
        ax.set_xlabel("Position")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Caterpillar plot
# ---------------------------------------------------------------------------


def caterpillar_plot(
    labels: Sequence[str],
    estimates: Sequence[float],
    ci_lower: Sequence[float],
    ci_upper: Sequence[float],
    *,
    sort: bool = True,
    title: str = "Caterpillar Plot",
    xlabel: str = "Estimate",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Caterpillar plot -- ranked effects with confidence intervals.

    Parameters
    ----------
    labels : sequence of str
    estimates : sequence of float
    ci_lower, ci_upper : sequence of float
    sort : bool
        Sort by estimate magnitude.
    title, xlabel : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(labels)
    estimates = np.asarray(estimates, dtype=float)
    ci_lower = np.asarray(ci_lower, dtype=float)
    ci_upper = np.asarray(ci_upper, dtype=float)
    n = len(labels)

    if sort:
        order = np.argsort(estimates)
        labels = [labels[i] for i in order]
        estimates = estimates[order]
        ci_lower = ci_lower[order]
        ci_upper = ci_upper[order]

    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * n)))
        else:
            fig = ax.figure

        y = np.arange(n)
        ax.hlines(y, ci_lower, ci_upper, color="#abd9e9", linewidth=2)
        ax.scatter(estimates, y, s=30, color="#2c7bb6", zorder=5, edgecolors="white", linewidths=0.5)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spaghetti plot
# ---------------------------------------------------------------------------


def spaghetti_plot(
    data: pd.DataFrame,
    time_col: str,
    value_col: str,
    id_col: str,
    *,
    group_col: str | None = None,
    max_lines: int = 100,
    show_mean: bool = True,
    title: str = "Individual Trajectories",
    theme: str | None = "publication",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Spaghetti plot showing individual trajectories over time.

    Parameters
    ----------
    data : DataFrame
        Long-format data.
    time_col : str
        Column with the time variable.
    value_col : str
        Column with the outcome.
    id_col : str
        Column identifying individuals.
    group_col : str or None
        Optional grouping variable for coloured trajectories.
    max_lines : int
        Maximum number of individual lines to draw.
    show_mean : bool
        Overlay the group/overall mean trajectory.
    title : str
    theme : str or None
    ax : Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    with _apply_theme_ctx(theme):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ids = data[id_col].unique()
        if len(ids) > max_lines:
            rng = np.random.default_rng(42)
            ids = rng.choice(ids, size=max_lines, replace=False)

        if group_col is not None:
            groups = data[group_col].unique()
            cmap = plt.get_cmap("Set1")
            color_map = {g: cmap(i) for i, g in enumerate(groups)}
        else:
            color_map = None

        for uid in ids:
            sub = data[data[id_col] == uid].sort_values(time_col)
            c = "#cccccc"
            if color_map is not None and group_col is not None:
                grp_vals = sub[group_col].unique()
                if len(grp_vals) > 0:
                    c = color_map.get(grp_vals[0], "#cccccc")
            ax.plot(sub[time_col], sub[value_col], color=c, alpha=0.3, linewidth=0.6)

        if show_mean:
            if group_col is not None:
                for g in groups:
                    g_data = data[data[group_col] == g]
                    mean_traj = g_data.groupby(time_col)[value_col].mean()
                    ax.plot(mean_traj.index, mean_traj.values, color=color_map[g], linewidth=2.5, label=str(g))
            else:
                mean_traj = data.groupby(time_col)[value_col].mean()
                ax.plot(mean_traj.index, mean_traj.values, color="#d7191c", linewidth=2.5, label="Mean")
            ax.legend(frameon=False, fontsize=8)

        ax.set_xlabel(time_col)
        ax.set_ylabel(value_col)
        ax.set_title(title)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    path: str,
    *,
    formats: list[str] | None = None,
    dpi: int = 300,
    close: bool = True,
) -> list[str]:
    """
    Save a matplotlib figure to one or more formats.

    Parameters
    ----------
    fig : Figure
        The figure to save.
    path : str
        Base path without extension (e.g. ``"output/forest"``).
    formats : list of str or None
        File extensions to save (default ``["pdf", "png"]``).
    dpi : int
        Resolution for raster formats.
    close : bool
        Whether to close the figure after saving.

    Returns
    -------
    list of str
        Paths to saved files.
    """
    if formats is None:
        formats = ["pdf", "png"]
    saved = []
    for fmt in formats:
        out = f"{path}.{fmt}"
        fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        saved.append(out)
        logger.info("Saved figure: %s", out)
    if close:
        plt.close(fig)
    return saved
