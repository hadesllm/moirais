#' Causal inference estimators for MORIE
#'
#' Implements ATE, ATT, ATC, GATE, CATE, and LATE via IPW, AIPW,
#' T-learner, and 2SLS. All estimators require propensity scores that
#' can be supplied or estimated internally via logistic regression.
#'
#' @name causal
#' @keywords internal
NULL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.fit_propensity <- function(data, treatment, covariates) {
  formula <- stats::as.formula(
    paste(treatment, "~", paste(covariates, collapse = " + "))
  )
  fit <- stats::glm(formula, data = data, family = stats::binomial())
  stats::fitted(fit)
}

.clip_ps <- function(ps, eps = 1e-6) {
  pmin(pmax(ps, eps), 1 - eps)
}

.hajek_diff <- function(y1, w1, y0, w0) {
  sum(y1 * w1) / sum(w1) - sum(y0 * w0) / sum(w0)
}

.influence_score_aipw <- function(y, t, ps, mu1, mu0) {
  (mu1 - mu0) +
    t * (y - mu1) / ps -
    (1 - t) * (y - mu0) / (1 - ps)
}


# ---------------------------------------------------------------------------
# Propensity scores
# ---------------------------------------------------------------------------

#' Estimate propensity scores via logistic regression
#'
#' @param data A data frame.
#' @param treatment Name of the binary treatment column.
#' @param covariates Character vector of covariate names.
#' @param trim Quantile pair used to winsorize extreme scores (default 0.01, 0.99).
#' @return Numeric vector of propensity scores (same length as `nrow(data)`).
#' @export
#' @examples
#' df <- data.frame(t = c(0,1,0,1,0,1), x = rnorm(6))
#' ps <- estimate_propensity_scores(df, "t", "x")
estimate_propensity_scores <- function(data, treatment, covariates,
                                       trim = c(0.01, 0.99)) {
  ps <- .fit_propensity(data, treatment, covariates)
  lo <- stats::quantile(ps, trim[1])
  hi <- stats::quantile(ps, trim[2])
  ps <- pmin(pmax(ps, lo), hi)
  .clip_ps(ps)
}


# ---------------------------------------------------------------------------
# ATE -- Hajek IPW
# ---------------------------------------------------------------------------

#' Estimate the Average Treatment Effect (ATE) via Hajek IPW
#'
#' The Hajek estimator uses stabilised IPW weights:
#' \deqn{\widehat{ATE} = \bar{y}_1^{w} - \bar{y}_0^{w}}
#' where \eqn{\bar{y}_t^{w} = \sum_{T_i=t} w_i Y_i / \sum_{T_i=t} w_i}
#' and \eqn{w_i = T_i/\hat{e}(X_i) + (1-T_i)/(1-\hat{e}(X_i))}.
#'
#' @param data A data frame.
#' @param treatment Name of the binary treatment column.
#' @param outcome Name of the outcome column.
#' @param covariates Character vector of covariate names.
#' @param propensity_col Optional: name of a pre-computed propensity score column.
#' @return Named list: `ate`, `se`, `ci_lower`, `ci_upper`, `n`, `ess`.
#' @export
#' @examples
#' set.seed(1)
#' df <- data.frame(
#'   t = rbinom(200, 1, 0.4),
#'   y = rnorm(200),
#'   x = rnorm(200)
#' )
#' estimate_ate(df, "t", "y", "x")
estimate_ate <- function(data, treatment, outcome, covariates,
                         propensity_col = NULL) {
  t <- as.numeric(data[[treatment]])
  y <- as.numeric(data[[outcome]])
  ps <- if (!is.null(propensity_col)) {
    .clip_ps(data[[propensity_col]])
  } else {
    estimate_propensity_scores(data, treatment, covariates)
  }

  w <- t / ps + (1 - t) / (1 - ps)
  ate <- .hajek_diff(y[t == 1], w[t == 1], y[t == 0], w[t == 0])
  se <- stats::sd(w * (t - ps) * y) / sqrt(length(y))
  ci <- .wald_ci(ate, se)
  ess <- (sum(w)^2) / sum(w^2)

  list(ate = ate, se = se, ci_lower = ci[1], ci_upper = ci[2],
       n = length(y), ess = ess)
}


# ---------------------------------------------------------------------------
# ATT -- Average Treatment Effect on the Treated
# ---------------------------------------------------------------------------

#' Estimate the Average Treatment Effect on the Treated (ATT)
#'
#' Treated units receive weight 1; controls receive
#' \eqn{w_i = \hat{e}(X_i)/(1-\hat{e}(X_i))}.
#'
#' @inheritParams estimate_ate
#' @return Named list: `att`, `se`, `ci_lower`, `ci_upper`, `n_treated`.
#' @export
#' @examples
#' set.seed(2)
#' df <- data.frame(t = rbinom(200, 1, 0.4), y = rnorm(200), x = rnorm(200))
#' estimate_att(df, "t", "y", "x")
estimate_att <- function(data, treatment, outcome, covariates,
                         propensity_col = NULL) {
  t <- as.numeric(data[[treatment]])
  y <- as.numeric(data[[outcome]])
  ps <- if (!is.null(propensity_col)) {
    .clip_ps(data[[propensity_col]])
  } else {
    estimate_propensity_scores(data, treatment, covariates)
  }

  # Control weights: e(X) / (1 - e(X))
  w_ctrl <- ps / (1 - ps)
  mean_t <- mean(y[t == 1])
  mean_c <- sum(y[t == 0] * w_ctrl[t == 0]) / sum(w_ctrl[t == 0])
  att <- mean_t - mean_c

  n1 <- sum(t == 1)
  # Delta-method SE approximation
  se <- sqrt(stats::var(y[t == 1]) / n1 +
             stats::var(w_ctrl[t == 0] * y[t == 0]) / sum(t == 0))
  ci <- .wald_ci(att, se)

  list(att = att, se = se, ci_lower = ci[1], ci_upper = ci[2], n_treated = n1)
}


# ---------------------------------------------------------------------------
# ATC -- Average Treatment Effect on the Controls
# ---------------------------------------------------------------------------

#' Estimate the Average Treatment Effect on the Controls (ATC)
#'
#' Control units receive weight 1; treated units receive
#' \eqn{w_i = (1-\hat{e}(X_i))/\hat{e}(X_i)}.
#'
#' @inheritParams estimate_ate
#' @return Named list: `atc`, `se`, `ci_lower`, `ci_upper`, `n_control`.
#' @export
estimate_atc <- function(data, treatment, outcome, covariates,
                         propensity_col = NULL) {
  t <- as.numeric(data[[treatment]])
  y <- as.numeric(data[[outcome]])
  ps <- if (!is.null(propensity_col)) {
    .clip_ps(data[[propensity_col]])
  } else {
    estimate_propensity_scores(data, treatment, covariates)
  }

  w_trt <- (1 - ps) / ps
  mean_treated_reweighted <- sum(y[t == 1] * w_trt[t == 1]) / sum(w_trt[t == 1])
  mean_c <- mean(y[t == 0])
  atc <- mean_treated_reweighted - mean_c

  n0 <- sum(t == 0)
  se <- sqrt(stats::var(y[t == 0]) / n0 +
             stats::var(w_trt[t == 1] * y[t == 1]) / sum(t == 1))
  ci <- .wald_ci(atc, se)

  list(atc = atc, se = se, ci_lower = ci[1], ci_upper = ci[2], n_control = n0)
}


# ---------------------------------------------------------------------------
# AIPW -- Doubly Robust ATE
# ---------------------------------------------------------------------------

#' Augmented IPW (AIPW) doubly-robust ATE estimator
#'
#' Combines IPW and outcome regression corrections. Consistent if
#' **either** the propensity model **or** the outcome model is correctly
#' specified.
#'
#' @inheritParams estimate_ate
#' @param outcome_model Family for the outcome model: `"linear"` or `"logistic"`.
#' @return Named list: `ate`, `se`, `ci_lower`, `ci_upper`, `n`.
#' @export
estimate_aipw <- function(data, treatment, outcome, covariates,
                          propensity_col = NULL,
                          outcome_model = c("linear", "logistic")) {
  outcome_model <- match.arg(outcome_model)
  t <- as.numeric(data[[treatment]])
  y <- as.numeric(data[[outcome]])
  ps <- if (!is.null(propensity_col)) {
    .clip_ps(data[[propensity_col]])
  } else {
    estimate_propensity_scores(data, treatment, covariates)
  }

  fam <- if (outcome_model == "logistic") stats::binomial() else stats::gaussian()
  formula <- stats::as.formula(
    paste(outcome, "~", paste(c(treatment, covariates), collapse = " + "))
  )
  fit <- stats::glm(formula, data = data, family = fam)
  data1 <- data; data1[[treatment]] <- 1
  data0 <- data; data0[[treatment]] <- 0
  mu1 <- as.numeric(stats::predict(fit, newdata = data1, type = "response"))
  mu0 <- as.numeric(stats::predict(fit, newdata = data0, type = "response"))

  psi <- .influence_score_aipw(y, t, ps, mu1, mu0)
  ate <- mean(psi)
  se <- stats::sd(psi) / sqrt(length(psi))
  ci <- .wald_ci(ate, se)

  list(ate = ate, se = se, ci_lower = ci[1], ci_upper = ci[2], n = length(y))
}


# ---------------------------------------------------------------------------
# GATE -- Group Average Treatment Effect
# ---------------------------------------------------------------------------

#' Estimate Group Average Treatment Effects (GATE)
#'
#' Applies AIPW within each level of `group_col` to estimate
#' stratum-specific treatment effects.
#'
#' @inheritParams estimate_aipw
#' @param group_col Name of the grouping variable (e.g. `"gender"`).
#' @return Data frame with columns: `group`, `ate`, `se`,
#'   `ci_lower`, `ci_upper`, `n`.
#' @export
#' @examples
#' set.seed(3)
#' df <- data.frame(
#'   t = rbinom(300, 1, 0.4),
#'   y = rnorm(300),
#'   x = rnorm(300),
#'   g = sample(c("A", "B"), 300, replace = TRUE)
#' )
#' estimate_gate(df, "t", "y", "x", "g")
estimate_gate <- function(data, treatment, outcome, covariates,
                          group_col, propensity_col = NULL,
                          outcome_model = c("linear", "logistic")) {
  outcome_model <- match.arg(outcome_model)
  groups <- unique(data[[group_col]])
  results <- vector("list", length(groups))

  for (i in seq_along(groups)) {
    g <- groups[i]
    sub <- data[data[[group_col]] == g, , drop = FALSE]
    if (nrow(sub) < 10 || length(unique(sub[[treatment]])) < 2) {
      results[[i]] <- data.frame(
        group = g, ate = NA_real_, se = NA_real_,
        ci_lower = NA_real_, ci_upper = NA_real_, n = nrow(sub)
      )
      next
    }
    est <- tryCatch(
      estimate_aipw(sub, treatment, outcome, covariates,
                    propensity_col = propensity_col,
                    outcome_model = outcome_model),
      error = function(e) list(ate = NA_real_, se = NA_real_,
                               ci_lower = NA_real_, ci_upper = NA_real_)
    )
    results[[i]] <- data.frame(
      group = g, ate = est$ate, se = est$se,
      ci_lower = est$ci_lower, ci_upper = est$ci_upper, n = nrow(sub)
    )
  }
  do.call(rbind, results)
}


# ---------------------------------------------------------------------------
# CATE -- Conditional (per-unit) treatment effects via T-learner
# ---------------------------------------------------------------------------

#' Estimate per-unit Conditional Average Treatment Effects (CATE)
#'
#' The **T-learner** fits separate outcome models on treated and control
#' units, then predicts the counterfactual for each unit:
#' \eqn{\widehat{CATE}_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)}.
#'
#' The **S-learner** fits one model with treatment as a feature.
#'
#' @inheritParams estimate_aipw
#' @param meta_learner `"t_learner"` (default) or `"s_learner"`.
#' @return Numeric vector of per-unit CATE estimates.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
estimate_cate <- function(data, treatment, outcome, covariates,
                          propensity_col = NULL,
                          outcome_model = c("linear", "logistic"),
                          meta_learner = c("t_learner", "s_learner")) {
  outcome_model <- match.arg(outcome_model)
  meta_learner  <- match.arg(meta_learner)
  fam <- if (outcome_model == "logistic") stats::binomial() else stats::gaussian()
  t <- as.numeric(data[[treatment]])

  rhs <- paste(covariates, collapse = " + ")
  formula <- stats::as.formula(paste(outcome, "~", rhs))

  if (meta_learner == "t_learner") {
    fit1 <- stats::glm(formula, data = data[t == 1, , drop = FALSE], family = fam)
    fit0 <- stats::glm(formula, data = data[t == 0, , drop = FALSE], family = fam)
    mu1 <- as.numeric(stats::predict(fit1, newdata = data, type = "response"))
    mu0 <- as.numeric(stats::predict(fit0, newdata = data, type = "response"))
  } else {
    formula_s <- stats::as.formula(
      paste(outcome, "~", paste(c(treatment, covariates), collapse = " + "))
    )
    fit <- stats::glm(formula_s, data = data, family = fam)
    data1 <- data; data1[[treatment]] <- 1
    data0 <- data; data0[[treatment]] <- 0
    mu1 <- as.numeric(stats::predict(fit, newdata = data1, type = "response"))
    mu0 <- as.numeric(stats::predict(fit, newdata = data0, type = "response"))
  }

  mu1 - mu0
}


# ---------------------------------------------------------------------------
# LATE -- Local Average Treatment Effect via 2SLS (Wald estimator)
# ---------------------------------------------------------------------------

#' Estimate the Local Average Treatment Effect (LATE) via 2SLS / Wald
#'
#' Uses a binary instrument \eqn{Z} to identify the LATE (Imbens & Angrist, 1994):
#' \deqn{LATE = \frac{Cov(Y, Z)}{Cov(T, Z)}}
#'
#' With covariates, uses two-stage OLS (Wald within residuals).
#' Requires `ivreg::ivreg()` if available; otherwise falls back to the
#' closed-form Wald estimator.
#'
#' @param data A data frame.
#' @param treatment Name of the binary endogenous treatment column.
#' @param outcome Name of the outcome column.
#' @param instrument Name of the binary instrument column.
#' @param covariates Optional character vector of exogenous covariates.
#' @return Named list: `late`, `se`, `ci_lower`, `ci_upper`,
#'   `first_stage_f`, `n`.
#' @export
#' @references
#'   Imbens GW, Angrist JD (1994). Identification and estimation of local
#'   average treatment effects. *Econometrica*, 62(2), 467-475.
estimate_late <- function(data, treatment, outcome, instrument,
                          covariates = NULL) {
  t <- as.numeric(data[[treatment]])
  y <- as.numeric(data[[outcome]])
  z <- as.numeric(data[[instrument]])

  # First-stage F statistic (strength of instrument)
  fs_formula <- stats::as.formula(
    paste(treatment, "~", instrument,
          if (!is.null(covariates)) paste("+", paste(covariates, collapse = " + ")) else "")
  )
  fs_fit <- stats::lm(fs_formula, data = data)
  fs_f <- summary(fs_fit)$fstatistic[1]

  # Wald estimator (no covariates)
  if (is.null(covariates)) {
    num <- stats::cov(y, z)
    den <- stats::cov(t, z)
    if (abs(den) < 1e-10) stop("Weak instrument: Cov(T, Z) ~= 0")
    late <- num / den
    # Delta-method SE
    n <- length(y)
    var_num <- stats::var(z * (y - late * t)) / n
    se <- sqrt(var_num) / abs(den)
  } else {
    # 2SLS via ivreg if available
    cov_str <- paste(covariates, collapse = " + ")
    iv_formula_str <- sprintf(
      "%s ~ %s + %s | %s + %s",
      outcome, treatment, cov_str, instrument, cov_str
    )
    if (requireNamespace("ivreg", quietly = TRUE)) {
      fit_iv <- ivreg::ivreg(
        stats::as.formula(iv_formula_str), data = data
      )
      late <- stats::coef(fit_iv)[treatment]
      se   <- sqrt(stats::vcov(fit_iv)[treatment, treatment])
    } else {
      # Fallback: manual 2SLS
      t_hat <- stats::fitted(fs_fit)
      data2 <- data
      data2[[paste0(treatment, "_hat")]] <- t_hat
      rhs2 <- paste(c(paste0(treatment, "_hat"), covariates), collapse = " + ")
      ss_fit <- stats::lm(
        stats::as.formula(paste(outcome, "~", rhs2)), data = data2
      )
      late <- stats::coef(ss_fit)[paste0(treatment, "_hat")]
      se   <- sqrt(stats::vcov(ss_fit)[paste0(treatment, "_hat"),
                                        paste0(treatment, "_hat")])
    }
  }

  ci <- .wald_ci(late, se)
  list(late = late, se = se, ci_lower = ci[1], ci_upper = ci[2],
       first_stage_f = as.numeric(fs_f), n = length(y))
}


# ---------------------------------------------------------------------------
# E-value (VanderWeele & Ding 2017)
# ---------------------------------------------------------------------------

#' Compute E-value for unmeasured confounding
#'
#' The E-value quantifies the minimum strength of confounding association
#' needed to fully explain away an observed treatment effect:
#' \deqn{E = RR + \sqrt{RR \cdot (RR - 1)}}
#'
#' For a risk ratio \eqn{RR < 1}, use \eqn{1/RR} before applying the formula.
#'
#' @param rr Risk ratio estimate (> 0). Supply > 1; if < 1, pass its reciprocal.
#' @param rr_lower Lower bound of the 95% CI (used to compute E-value for CI).
#' @return Named list: `e_value`, `e_value_ci` (for the CI bound).
#' @export
#' @references
#'   VanderWeele TJ, Ding P (2017). Sensitivity analysis in observational
#'   research: introducing the E-value. *Annals of Internal Medicine*,
#'   167(4):268-274.
#' @examples
#' e_value(rr = 3.9, rr_lower = 2.4)
e_value <- function(rr, rr_lower = NULL) {
  compute_e <- function(r) r + sqrt(r * (r - 1))
  ev <- compute_e(rr)
  ev_ci <- if (!is.null(rr_lower)) compute_e(rr_lower) else NA_real_
  list(e_value = ev, e_value_ci = ev_ci)
}


# ---------------------------------------------------------------------------
# Sensitivity analysis -- Rosenbaum bounds
# ---------------------------------------------------------------------------

#' Rosenbaum bounds sensitivity analysis
#'
#' For a range of hidden-confounding levels \eqn{\Gamma}, tests whether
#' the treatment effect remains significant. A large \eqn{\Gamma} at
#' which the result remains significant indicates robustness.
#'
#' Uses Wilcoxon signed-rank statistic bounds for matched designs.
#' For unmatched data, computes sign-score bounds.
#'
#' @param treated Numeric vector of outcomes for treated units.
#' @param control Numeric vector of outcomes for control units
#'   (may differ in length from `treated` for unmatched designs).
#' @param gamma_range Numeric vector of \eqn{\Gamma} values to test.
#' @return Data frame with columns: `gamma`, `p_lower`, `p_upper`.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
#' @references
#'   Rosenbaum PR (2002). *Observational Studies* (2nd ed.). Springer.
sensitivity_rosenbaum <- function(treated, control,
                                  gamma_range = seq(1, 3, by = 0.2)) {
  n1 <- length(treated)
  n0 <- length(control)

  results <- lapply(gamma_range, function(gamma) {
    # Sign score bounds (Rosenbaum 2002, Section 4.3)
    # Under null, each unit has outcome contribution +/-1
    y_diff <- outer(treated, control, "-")
    signs <- sign(y_diff)
    n_pairs <- n1 * n0

    # Upper bound: p-value under maximum assignment probability
    p_plus  <- gamma / (1 + gamma)
    p_minus <- 1 / (1 + gamma)

    # Expected value and variance under gamma
    E_upper <- sum(p_plus  * (signs > 0) + p_minus * (signs < 0))
    E_lower <- sum(p_minus * (signs > 0) + p_plus  * (signs < 0))
    V <- n_pairs * p_plus * p_minus

    T_stat <- sum(signs > 0)
    p_upper <- 1 - stats::pnorm((T_stat - E_lower) / sqrt(V))
    p_lower <- 1 - stats::pnorm((T_stat - E_upper) / sqrt(V))

    data.frame(gamma = gamma, p_lower = p_lower, p_upper = p_upper)
  })

  do.call(rbind, results)
}


# ---------------------------------------------------------------------------
# G-computation (outcome regression ATE)
# ---------------------------------------------------------------------------

#' G-computation (outcome regression) ATE estimator
#'
#' Estimates the ATE by:
#' \deqn{\widehat{ATE} = \frac{1}{n}\sum_i \bigl[\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)\bigr]}
#'
#' @inheritParams estimate_aipw
#' @return Named list: `ate`, `se`, `ci_lower`, `ci_upper`.
#' @export
estimate_g_computation <- function(data, treatment, outcome, covariates,
                                   outcome_model = c("linear", "logistic")) {
  outcome_model <- match.arg(outcome_model)
  fam <- if (outcome_model == "logistic") stats::binomial() else stats::gaussian()
  formula <- stats::as.formula(
    paste(outcome, "~", paste(c(treatment, covariates), collapse = " + "))
  )
  fit <- stats::glm(formula, data = data, family = fam)
  data1 <- data; data1[[treatment]] <- 1
  data0 <- data; data0[[treatment]] <- 0
  mu1 <- as.numeric(stats::predict(fit, newdata = data1, type = "response"))
  mu0 <- as.numeric(stats::predict(fit, newdata = data0, type = "response"))
  diffs <- mu1 - mu0
  ate <- mean(diffs)
  se <- stats::sd(diffs) / sqrt(length(diffs))
  ci <- .wald_ci(ate, se)
  list(ate = ate, se = se, ci_lower = ci[1], ci_upper = ci[2])
}
