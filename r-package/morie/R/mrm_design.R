# SPDX-License-Identifier: GPL-2.0-only

#' Experimental-design callables (designexptr-inspired)
#'
#' R parity of \code{morie.mrm_design} (Python).  Four general-
#' purpose statistical-design entry points covering the
#' designexptr.org pedagogical sequence: two-treatment comparison,
#' one-way ANOVA with Tukey HSD, 2^k factorial design, and a
#' designed-experiment convenience wrapper around the morie causal
#' estimator family.
#'
#' @references
#' Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005).
#'   Statistics for Experimenters. Wiley.
#'
#' @name mrm_design
NULL


#' Two-treatment outcome comparison with three assumption regimes
#'
#' Always returns Welch t (unequal variance, canonical), Student t
#' (equal variance), and Mann-Whitney U (rank-based).  The Welch
#' p-value is the canonical answer; the others are the sensitivity
#' range.
#'
#' @param a,b Outcome vectors under treatments A and B.
#' @param alpha CI level (default 0.05).
#' @return Named list with estimate, se, t_statistic, df,
#'   p_welch, p_student, p_mannwhitney, ci_lower, ci_upper,
#'   n_a, n_b, interpretation.
#' @export
mrm_two_treatment_test <- function(a, b, alpha = 0.05) {
  a <- as.numeric(a); b <- as.numeric(b)
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  welch <- stats::t.test(a, b, var.equal = FALSE)
  stud  <- stats::t.test(a, b, var.equal = TRUE)
  mw    <- suppressWarnings(stats::wilcox.test(a, b, alternative = "two.sided"))
  diff  <- mean(a) - mean(b)
  se    <- sqrt(stats::var(a) / length(a) + stats::var(b) / length(b))
  df_   <- welch$parameter
  z     <- stats::qt(1 - alpha / 2, df_)
  list(
    estimate     = round(diff, 6),
    se           = round(se, 6),
    t_statistic  = round(as.numeric(welch$statistic), 4),
    df           = round(as.numeric(df_), 2),
    p_welch      = welch$p.value,
    p_student    = stud$p.value,
    p_mannwhitney= mw$p.value,
    ci_lower     = round(diff - z * se, 6),
    ci_upper     = round(diff + z * se, 6),
    n_a          = length(a), n_b = length(b),
    interpretation = sprintf("Welch t: Delta=%.3f, p=%.3g; %s",
                             diff, welch$p.value,
                             if (welch$p.value < alpha) "reject H0" else "fail to reject")
  )
}


#' One-way ANOVA + Tukey HSD post-hoc
#'
#' @param data data.frame containing response and group columns.
#' @param response_col,group_col Column names.
#' @param alpha CI level (default 0.05).
#' @return Named list with f_statistic, p_value, df_between,
#'   df_within, means, n_per_group, tukey_hsd, interpretation.
#' @export
mrm_anova_oneway <- function(data, response_col, group_col, alpha = 0.05) {
  d <- data[, c(response_col, group_col)]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  d[[group_col]] <- factor(d[[group_col]])
  fit <- stats::aov(as.formula(paste(response_col, "~", group_col)), data = d)
  summ <- summary(fit)[[1]]
  f <- summ[["F value"]][1]; p <- summ[["Pr(>F)"]][1]
  tk <- stats::TukeyHSD(fit, conf.level = 1 - alpha)
  tk_df <- as.data.frame(tk[[group_col]])
  tk_df$pair <- rownames(tk_df); rownames(tk_df) <- NULL
  means <- tapply(d[[response_col]], d[[group_col]], mean)
  ns    <- tapply(d[[response_col]], d[[group_col]], length)
  list(
    f_statistic = round(as.numeric(f), 4),
    p_value     = as.numeric(p),
    df_between  = summ[["Df"]][1],
    df_within   = summ[["Df"]][2],
    means       = as.list(round(means, 4)),
    n_per_group = as.list(ns),
    tukey_hsd   = tk_df,
    interpretation = sprintf("F(%d,%d) = %.3f, p = %.3g%s",
                             summ[["Df"]][1], summ[["Df"]][2], f, p,
                             if (p < alpha) "; reject H0 of equal means" else "")
  )
}


#' 2^k factorial-design analysis with main effects + interactions
#'
#' Returns main effects (difference of means at +1 vs -1 per factor),
#' all interaction effects, and half-normal-plot coordinates for
#' Daniel's method (which lets the user separate active effects from
#' a null half-normal line on the same axes).
#'
#' @param data data.frame with response_col and factor_cols.  Factor
#'   columns may be coded as \code{+/- 1} or any binary; non-(-1,1)
#'   columns are re-coded.
#' @param response_col Numeric response column.
#' @param factor_cols Character vector of factor column names.
#' @return Named list with main_effects, interaction_effects,
#'   half_normal_coords (data.frame), n, k, interpretation.
#' @export
mrm_factorial_2k <- function(data, response_col, factor_cols) {
  k <- length(factor_cols)
  if (k < 2L) stop("2^k design requires k >= 2 factors")
  d <- data[, c(response_col, factor_cols), drop = FALSE]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  # Re-code each factor to +/-1
  for (c in factor_cols) {
    v <- d[[c]]
    if (!all(unique(v) %in% c(-1, 1))) {
      d[[c]] <- ifelse(v > stats::median(v), 1L, -1L)
    }
  }
  y <- d[[response_col]]
  X <- as.matrix(d[, factor_cols, drop = FALSE])
  main <- vapply(seq_along(factor_cols), function(i) {
    pos <- X[, i] == 1
    mean(y[pos]) - mean(y[!pos])
  }, numeric(1))
  names(main) <- factor_cols

  # Interactions of order 2..k
  inter <- list()
  for (r in 2L:k) {
    for (cb in utils::combn(seq_along(factor_cols), r, simplify = FALSE)) {
      nm <- paste(factor_cols[cb], collapse = " x ")
      prod <- apply(X[, cb, drop = FALSE], 1, prod)
      inter[[nm]] <- mean(y[prod == 1]) - mean(y[prod == -1])
    }
  }

  all_eff <- c(as.list(main), inter)
  mags <- sort(abs(unlist(all_eff)))
  n_eff <- length(mags)
  half_n <- data.frame(
    effect_name = names(mags),
    effect_magnitude = round(mags, 6),
    quantile = ((seq_len(n_eff) - 0.5) / n_eff),
    half_normal_quantile = stats::qnorm(0.5 + 0.5 * (seq_len(n_eff) - 0.5) / n_eff),
    stringsAsFactors = FALSE
  )

  list(
    main_effects = as.list(round(main, 6)),
    interaction_effects = lapply(inter, function(x) round(x, 6)),
    half_normal_coords = half_n,
    n = length(y), k = k,
    interpretation = sprintf("2^%d factorial on n=%d. Largest main effect: %s = %.3f",
                             k, length(y),
                             names(main)[which.max(abs(main))],
                             main[which.max(abs(main))])
  )
}


#' Designed-experiment convenience wrapper around the morie causal
#' estimator family
#'
#' @param data data.frame with treatment, outcome, optional
#'   covariates.
#' @param treatment_col Binary 0/1 treatment column.
#' @param outcome_col Continuous outcome column.
#' @param covariates Optional character vector of covariate columns.
#' @param estimator One of \code{"ipw"} (Hajek IPW with logistic
#'   propensity), \code{"diff_in_means"} (no adjustment).
#' @return Named list with estimator, estimate, se, ci_lower,
#'   ci_upper, p_value, n, n_treated, interpretation.
#' @export
mrm_causal_design <- function(
  data, treatment_col, outcome_col,
  covariates = character(0),
  estimator = c("ipw", "diff_in_means")
) {
  estimator <- match.arg(estimator)
  cols <- c(treatment_col, outcome_col, covariates)
  d <- data[, cols, drop = FALSE]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  D <- as.integer(d[[treatment_col]])
  Y <- as.numeric(d[[outcome_col]])
  n <- nrow(d); n_t <- sum(D)

  if (estimator == "ipw" && length(covariates) > 0L) {
    fml <- as.formula(paste(treatment_col, "~", paste(covariates, collapse = "+")))
    fit <- stats::glm(fml, data = d, family = stats::binomial())
    e   <- stats::predict(fit, type = "response")
    e   <- pmax(pmin(e, 1 - 1e-6), 1e-6)
    w1  <- D / e; w0 <- (1 - D) / (1 - e)
    tau <- sum(w1 * Y) / sum(w1) - sum(w0 * Y) / sum(w0)
    # bootstrap SE
    set.seed(42)
    boots <- replicate(199, {
      idx <- sample.int(n, replace = TRUE)
      sub <- d[idx, , drop = FALSE]
      e_b <- tryCatch({
        fit_b <- stats::glm(fml, data = sub, family = stats::binomial())
        pmax(pmin(stats::predict(fit_b, type = "response"), 1 - 1e-6), 1e-6)
      }, error = function(e) NA_real_)
      if (all(is.na(e_b))) NA_real_ else {
        Db <- as.integer(sub[[treatment_col]]); Yb <- as.numeric(sub[[outcome_col]])
        w1b <- Db / e_b; w0b <- (1 - Db) / (1 - e_b)
        sum(w1b * Yb) / sum(w1b) - sum(w0b * Yb) / sum(w0b)
      }
    })
    se <- stats::sd(boots, na.rm = TRUE)
  } else {
    tau <- mean(Y[D == 1]) - mean(Y[D == 0])
    se  <- sqrt(stats::var(Y[D == 1]) / sum(D == 1) +
                stats::var(Y[D == 0]) / sum(D == 0))
  }

  z <- 1.96
  list(
    estimator = estimator,
    estimate = round(tau, 6), se = round(se, 6),
    ci_lower = round(tau - z * se, 6),
    ci_upper = round(tau + z * se, 6),
    p_value  = 2 * (1 - stats::pnorm(abs(tau / se))),
    n = n, n_treated = as.integer(n_t),
    interpretation = sprintf("%s ATE = %.4f (SE %.4f); 95%% CI [%.4f, %.4f]",
                             toupper(estimator), tau, se,
                             tau - z * se, tau + z * se)
  )
}
