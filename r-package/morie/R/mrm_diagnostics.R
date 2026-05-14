# SPDX-License-Identifier: MIT OR Apache-2.0

#' Causal-inference diagnostics (R parity)
#'
#' Balance, overlap, SUTVA-style assumption checks, and the median
#' causal effect estimator.  R parity of \code{morie.mrm_diagnostics}.
#'
#' @references
#' Imbens, G. W., & Rubin, D. B. (2015). Causal Inference for Statistics,
#'   Social and Biomedical Sciences. Cambridge University Press.
#' Rosenbaum, P. R., & Rubin, D. B. (1985). Constructing a control group
#'   using multivariate matched sampling methods that incorporate the
#'   propensity score. The American Statistician, 39(1), 33-38.
#' Cole, S. R., & Hernan, M. A. (2008). Constructing inverse probability
#'   weights for marginal structural models. AJE, 168(6), 656-664.
#'
#' @name mrm_diagnostics
NULL


.morie_logistic_propensity <- function(D, X) {
  d <- data.frame(D = D, X)
  fit <- suppressWarnings(stats::glm(D ~ ., data = d, family = stats::binomial()))
  e <- stats::predict(fit, type = "response")
  pmax(pmin(e, 1 - 1e-6), 1e-6)
}


#' Imbens-Rubin standardised %SMD per covariate
#'
#' For continuous X:  SMD = (mean_t - mean_c) / sqrt((s2_t + s2_c)/2)
#' For binary X:      SMD = (p_t - p_c) / sqrt((p_t(1-p_t) + p_c(1-p_c))/2)
#' Returned as percent.  |SMD| > 10 is the Austin (2009) imbalance threshold.
#'
#' @param data data.frame.
#' @param treatment_col Binary 0/1 treatment column name.
#' @param covariates Character vector of covariate columns.
#' @return data.frame with covariate, mean_treated, mean_control,
#'   pooled_sd, smd_pct, imbalanced columns.
#' @examples
#' set.seed(2026)
#' n <- 200L
#' df <- data.frame(
#'   D   = rbinom(n, 1, 0.4),
#'   age = rnorm(n, 50, 10),
#'   bmi = rnorm(n, 27, 4)
#' )
#' df$age[df$D == 1] <- df$age[df$D == 1] + 3   # deliberate imbalance
#' mrm_standardised_difference(df, treatment_col = "D",
#'                             covariates = c("age", "bmi"))
#' @export
mrm_standardised_difference <- function(data, treatment_col, covariates) {
  D <- as.integer(data[[treatment_col]])
  out <- lapply(covariates, function(c) {
    x   <- as.numeric(data[[c]])
    x_t <- x[D == 1]; x_c <- x[D == 0]
    m_t <- mean(x_t); m_c <- mean(x_c)
    s_t <- stats::var(x_t); s_c <- stats::var(x_c)
    pooled_sd <- sqrt((s_t + s_c) / 2)
    smd_pct <- if (pooled_sd > 0) 100 * (m_t - m_c) / pooled_sd else NA_real_
    data.frame(
      covariate    = c,
      mean_treated = round(m_t, 4),
      mean_control = round(m_c, 4),
      pooled_sd    = round(pooled_sd, 4),
      smd_pct      = round(smd_pct, 2),
      imbalanced   = if (is.na(smd_pct)) NA else abs(smd_pct) > 10,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}


#' Composite balance verdict using the Imbens-Rubin %SMD criterion
#'
#' A design is "balanced on X" if every |SMD(X_i)| <= threshold.
#'
#' @param data data.frame.
#' @param treatment_col Binary 0/1 treatment column.
#' @param covariates Character vector of covariate columns.
#' @param threshold_pct %SMD imbalance threshold (default 10).
#' @return Named list with table, threshold_pct, n_imbalanced,
#'   overall_balanced, interpretation.
#' @examples
#' set.seed(2026)
#' n <- 200L
#' df <- data.frame(
#'   D   = rbinom(n, 1, 0.4),
#'   age = rnorm(n, 50, 10),
#'   bmi = rnorm(n, 27, 4)
#' )
#' df$age[df$D == 1] <- df$age[df$D == 1] + 3   # imbalance on age
#' bal <- mrm_check_balancing(df, treatment_col = "D",
#'                            covariates = c("age", "bmi"))
#' bal$overall_balanced
#' bal$interpretation
#' @export
mrm_check_balancing <- function(data, treatment_col, covariates,
                                 threshold_pct = 10) {
  tbl <- mrm_standardised_difference(data, treatment_col, covariates)
  n_imb <- sum(tbl$imbalanced, na.rm = TRUE)
  overall <- n_imb == 0L
  list(
    table = tbl,
    threshold_pct = threshold_pct,
    n_imbalanced = as.integer(n_imb),
    overall_balanced = overall,
    interpretation = sprintf(
      "%d/%d covariates exceed |SMD|>%g%%; design %s on this covariate set.",
      n_imb, length(covariates), threshold_pct,
      if (overall) "BALANCED" else "UNBALANCED"
    )
  )
}


#' Propensity-score support overlap diagnostic (Cole-Hernan 2008)
#'
#' @param data data.frame.
#' @param treatment_col Binary 0/1 treatment column.
#' @param covariates Character vector of covariates.
#' @return Named list with e_treated_quantiles, e_control_quantiles,
#'   common_support_lower, common_support_upper, n_outside_support,
#'   positivity_violations, interpretation.
#' @examples
#' set.seed(2026)
#' n <- 300L
#' x <- rnorm(n)
#' D <- rbinom(n, 1, plogis(0.5 * x))
#' df <- data.frame(D = D, age = x)
#' ovl <- mrm_check_overlap(df, treatment_col = "D",
#'                          covariates = "age")
#' ovl$positivity_violations
#' ovl$interpretation
#' @export
mrm_check_overlap <- function(data, treatment_col, covariates) {
  D <- as.integer(data[[treatment_col]])
  X <- as.data.frame(data[, covariates, drop = FALSE])
  e <- .morie_logistic_propensity(D, X)
  e_t <- e[D == 1]; e_c <- e[D == 0]
  qs  <- c(0.025, 0.25, 0.5, 0.75, 0.975)
  cs_lo <- max(min(e_t), min(e_c))
  cs_hi <- min(max(e_t), max(e_c))
  n_outside <- sum(e < cs_lo | e > cs_hi)
  pviol <- sum(e < 0.01 | e > 0.99)
  name_q <- function(q) paste0("q", q * 100)
  list(
    e_treated_quantiles = setNames(as.list(round(stats::quantile(e_t, qs), 4)),
                                    name_q(qs)),
    e_control_quantiles = setNames(as.list(round(stats::quantile(e_c, qs), 4)),
                                    name_q(qs)),
    common_support_lower = round(cs_lo, 4),
    common_support_upper = round(cs_hi, 4),
    n_outside_support = as.integer(n_outside),
    positivity_violations = as.integer(pviol),
    interpretation = sprintf(
      "common support [%.3f, %.3f]; %d units outside; %d positivity violations (e<.01 or e>.99).",
      cs_lo, cs_hi, n_outside, pviol
    )
  )
}


#' Median causal effect via 1:1 nearest-neighbour PS matching
#'
#' @param data data.frame.
#' @param treatment_col Binary 0/1 column.
#' @param outcome_col Outcome column name.
#' @param covariates Character vector of covariate columns.
#' @return Named list with median_y1, median_y0,
#'   median_treatment_effect, n_matched, interpretation.
#' @examples
#' set.seed(2026)
#' n <- 200L
#' x <- rnorm(n)
#' D <- rbinom(n, 1, plogis(0.5 * x))
#' y <- 0.7 * D + 0.3 * x + rnorm(n, 0, 0.5)
#' df <- data.frame(D = D, y = y, age = x)
#' res <- mrm_median_causal_effect(df, treatment_col = "D",
#'                                 outcome_col = "y",
#'                                 covariates = "age")
#' res$median_treatment_effect
#' res$n_matched
#' @export
mrm_median_causal_effect <- function(data, treatment_col, outcome_col,
                                      covariates) {
  D <- as.integer(data[[treatment_col]])
  Y <- as.numeric(data[[outcome_col]])
  X <- as.data.frame(data[, covariates, drop = FALSE])
  e <- .morie_logistic_propensity(D, X)
  logit <- log(e / (1 - e))
  treated_idx <- which(D == 1L)
  control_idx <- which(D == 0L)
  used <- integer(0)
  pairs <- vector("list", length(treated_idx))
  k <- 0L
  for (i in treated_idx) {
    avail <- setdiff(control_idx, used)
    if (!length(avail)) break
    dists <- abs(logit[avail] - logit[i])
    j <- avail[which.min(dists)]
    used <- c(used, j); k <- k + 1L
    pairs[[k]] <- c(i, j)
  }
  pairs <- pairs[seq_len(k)]
  if (!length(pairs)) stop("no valid matches")
  Y1 <- vapply(pairs, function(p) Y[p[1]], numeric(1))
  Y0 <- vapply(pairs, function(p) Y[p[2]], numeric(1))
  m1 <- stats::median(Y1); m0 <- stats::median(Y0)
  list(
    median_y1 = round(m1, 4),
    median_y0 = round(m0, 4),
    median_treatment_effect = round(m1 - m0, 4),
    n_matched = as.integer(length(pairs)),
    interpretation = sprintf(
      "Median Y(1) = %.3f, Median Y(0) = %.3f; median causal effect = %.3f on %d 1:1 PS-matched pairs.",
      m1, m0, m1 - m0, length(pairs)
    )
  )
}


#' Composite Rubin-style identifiability assumption check
#'
#' Returns each assumption with diagnostic evidence + a flag.
#'
#' @param data data.frame.
#' @param treatment_col Binary 0/1 column.
#' @param outcome_col Outcome column (presently unused; reserved
#'   for future E-value evidence).
#' @param covariates Character vector of covariate columns.
#' @return Named list with sutva, unconfoundedness,
#'   probabilistic_assignment, overall_verdict sub-lists.
#' @examples
#' set.seed(2026)
#' n <- 300L
#' x <- rnorm(n)
#' D <- rbinom(n, 1, plogis(0.5 * x))
#' y <- 0.7 * D + 0.3 * x + rnorm(n)
#' df <- data.frame(D = D, y = y, age = x)
#' chk <- mrm_assumptions_check(df, treatment_col = "D",
#'                              outcome_col = "y",
#'                              covariates = "age")
#' chk$overall_verdict
#' @export
mrm_assumptions_check <- function(data, treatment_col, outcome_col,
                                   covariates) {
  overlap <- mrm_check_overlap(data, treatment_col, covariates)
  balance <- mrm_check_balancing(data, treatment_col, covariates)
  sutva <- list(
    verdict = "untestable from data",
    evidence = paste(
      "SUTVA is a design assumption; reviewer should justify it from",
      "context (e.g. unit-level non-interference, single treatment",
      "definition). Within-cluster variance check is not run here."
    )
  )
  unconf <- list(
    verdict = if (balance$overall_balanced) {
      "plausible (after adjustment)"
    } else {
      "questionable - covariate imbalance remains"
    },
    evidence = balance$interpretation
  )
  pos <- list(
    verdict = if (overlap$positivity_violations == 0L) "satisfied" else "violated",
    evidence = sprintf(
      "common support [%g, %g]; %d units with e(x) outside (0.01, 0.99).",
      overlap$common_support_lower, overlap$common_support_upper,
      overlap$positivity_violations
    )
  )
  overall <- if (balance$overall_balanced &&
                  overlap$positivity_violations == 0L) {
    "all three assumptions ok modulo SUTVA design-context"
  } else {
    "one or more diagnostic flags; see fields"
  }
  list(
    sutva = sutva,
    unconfoundedness = unconf,
    probabilistic_assignment = pos,
    overall_verdict = overall
  )
}
