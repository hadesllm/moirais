#' Frequentist inference helpers for MOIRAIS
#'
#' Confidence intervals, hypothesis tests, effect sizes, and power analysis.
#' All distribution functions follow R's standard naming convention
#' (`dnorm`, `pnorm`, `qnorm`, `rnorm`, etc.) and are re-exported with
#' additional MOIRAIS-specific wrappers for epidemiological use cases.
#'
#' @name inference
#' @keywords internal
NULL


# ---------------------------------------------------------------------------
# Two-sample comparison helpers
# ---------------------------------------------------------------------------

#' Two-sample t-test with tidy output
#'
#' @param x1 Numeric vector (group 1).
#' @param x2 Numeric vector (group 2).
#' @param equal_var Assume equal variances? Default `FALSE` (Welch test).
#' @param alternative `"two.sided"`, `"greater"`, or `"less"`.
#' @return Named list: `t`, `df`, `p_value`, `ci_diff`, `cohens_d`.
#' @export
#' @examples
#' two_sample_t_test(rnorm(50, 0.5), rnorm(50, 0))
two_sample_t_test <- function(x1, x2,
                               equal_var = FALSE,
                               alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)
  result <- stats::t.test(x1, x2, var.equal = equal_var,
                          alternative = alternative)
  d <- cohens_d(x1, x2)
  list(
    t = as.numeric(result$statistic),
    df = as.numeric(result$parameter),
    p_value = result$p.value,
    ci_diff = as.numeric(result$conf.int),
    cohens_d = d
  )
}

#' One-sample t-test
#'
#' @param x Numeric vector.
#' @param mu0 Null hypothesis mean (default 0).
#' @param alternative `"two.sided"`, `"greater"`, or `"less"`.
#' @return Named list: `t`, `df`, `p_value`, `ci`.
#' @export
one_sample_t_test <- function(x, mu0 = 0,
                               alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)
  result <- stats::t.test(x, mu = mu0, alternative = alternative)
  list(
    t = as.numeric(result$statistic),
    df = as.numeric(result$parameter),
    p_value = result$p.value,
    ci = as.numeric(result$conf.int)
  )
}

#' Paired t-test
#'
#' @param x1 Numeric vector (before/condition 1).
#' @param x2 Numeric vector (after/condition 2).
#' @param alternative `"two.sided"`, `"greater"`, or `"less"`.
#' @return Named list: `t`, `df`, `p_value`, `ci_diff`, `mean_diff`.
#' @export
paired_t_test <- function(x1, x2,
                           alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)
  result <- stats::t.test(x1, x2, paired = TRUE, alternative = alternative)
  list(
    t = as.numeric(result$statistic),
    df = as.numeric(result$parameter),
    p_value = result$p.value,
    ci_diff = as.numeric(result$conf.int),
    mean_diff = mean(x1 - x2, na.rm = TRUE)
  )
}

#' Chi-square test of independence or goodness-of-fit
#'
#' @param observed Observed counts (matrix for independence, vector for GOF).
#' @param expected Expected counts for GOF (optional; uniform if NULL).
#' @return Named list: `chi_sq`, `df`, `p_value`, `cramers_v`.
#' @export
chi_square_test <- function(observed, expected = NULL) {
  if (is.matrix(observed) || is.data.frame(observed)) {
    result <- stats::chisq.test(observed)
    v <- cramers_v(as.matrix(observed))
  } else {
    result <- stats::chisq.test(observed, p = expected)
    v <- NA_real_
  }
  list(
    chi_sq = as.numeric(result$statistic),
    df = as.numeric(result$parameter),
    p_value = result$p.value,
    cramers_v = v
  )
}

#' Fisher's exact test for 2×2 tables
#'
#' @param table_2x2 A 2×2 matrix or data frame of counts.
#' @param alternative `"two.sided"`, `"greater"`, or `"less"`.
#' @return Named list: `odds_ratio`, `ci`, `p_value`.
#' @export
fisher_exact_test <- function(table_2x2,
                               alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)
  result <- stats::fisher.test(as.matrix(table_2x2), alternative = alternative)
  list(
    odds_ratio = as.numeric(result$estimate),
    ci = as.numeric(result$conf.int),
    p_value = result$p.value
  )
}

#' One-way ANOVA
#'
#' @param ... Numeric vectors, one per group.
#' @return Named list: `F`, `df_between`, `df_within`, `p_value`,
#'   `eta_squared`.
#' @export
#' @examples
#' anova_one_way(rnorm(30, 0), rnorm(30, 0.5), rnorm(30, 1))
anova_one_way <- function(...) {
  groups <- list(...)
  if (length(groups) < 2) stop("At least two groups required.")
  df_long <- do.call(rbind, lapply(seq_along(groups), function(i) {
    data.frame(y = groups[[i]], grp = factor(i))
  }))
  fit <- stats::aov(y ~ grp, data = df_long)
  s <- summary(fit)[[1]]
  f_val <- s["grp", "F value"]
  df_b  <- s["grp", "Df"]
  df_w  <- s["Residuals", "Df"]
  ss_b  <- s["grp", "Sum Sq"]
  ss_t  <- sum(s[, "Sum Sq"])
  list(
    F = f_val,
    df_between = df_b,
    df_within  = df_w,
    p_value    = s["grp", "Pr(>F)"],
    eta_squared = ss_b / ss_t
  )
}

#' Kruskal-Wallis non-parametric ANOVA
#'
#' @param ... Numeric vectors, one per group.
#' @return Named list: `H`, `df`, `p_value`.
#' @export
kruskal_wallis_test <- function(...) {
  groups <- list(...)
  df_long <- do.call(rbind, lapply(seq_along(groups), function(i) {
    data.frame(y = groups[[i]], grp = factor(i))
  }))
  result <- stats::kruskal.test(y ~ grp, data = df_long)
  list(
    H = as.numeric(result$statistic),
    df = as.numeric(result$parameter),
    p_value = result$p.value
  )
}

#' Mann-Whitney U test (Wilcoxon rank-sum)
#'
#' @param x1 Numeric vector (group 1).
#' @param x2 Numeric vector (group 2).
#' @param alternative `"two.sided"`, `"greater"`, or `"less"`.
#' @return Named list: `W`, `p_value`, `r` (effect size).
#' @export
mann_whitney_test <- function(x1, x2,
                               alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)
  result <- stats::wilcox.test(x1, x2, alternative = alternative,
                                exact = FALSE)
  n <- length(x1) * length(x2)
  r_effect <- abs(stats::qnorm(result$p.value / 2)) / sqrt(n)
  list(W = as.numeric(result$statistic), p_value = result$p.value, r = r_effect)
}

#' Wilcoxon signed-rank test (paired)
#'
#' @param x1 Numeric vector (before).
#' @param x2 Numeric vector (after).
#' @param alternative `"two.sided"`, `"greater"`, or `"less"`.
#' @return Named list: `V`, `p_value`.
#' @export
wilcoxon_signed_rank_test <- function(x1, x2,
                                       alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)
  result <- stats::wilcox.test(x1, x2, paired = TRUE,
                                alternative = alternative, exact = FALSE)
  list(V = as.numeric(result$statistic), p_value = result$p.value)
}

#' Shapiro-Wilk normality test
#'
#' @param x Numeric vector.
#' @param alpha Significance level for the `is_normal` flag (default 0.05).
#' @return Named list: `W`, `p_value`, `is_normal`.
#' @export
shapiro_wilk_test <- function(x, alpha = 0.05) {
  result <- stats::shapiro.test(x)
  list(
    W = as.numeric(result$statistic),
    p_value = result$p.value,
    is_normal = result$p.value > alpha
  )
}

#' Levene test for equality of variances
#'
#' @param ... Numeric vectors, one per group.
#' @return Named list: `F`, `p_value`.
#' @export
levene_test <- function(...) {
  groups <- list(...)
  df_long <- do.call(rbind, lapply(seq_along(groups), function(i) {
    data.frame(y = groups[[i]], grp = factor(i))
  }))
  # Levene statistic via absolute deviations from group medians
  df_long$dev <- abs(df_long$y - ave(df_long$y, df_long$grp, FUN = median))
  fit <- stats::aov(dev ~ grp, data = df_long)
  s <- summary(fit)[[1]]
  list(F = s["grp", "F value"], p_value = s["grp", "Pr(>F)"])
}


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

#' Wilson score confidence interval for a proportion
#'
#' @param successes Number of successes.
#' @param n Total observations.
#' @param alpha Significance level (default 0.05 → 95% CI).
#' @param method `"wilson"` (default), `"exact"` (Clopper-Pearson),
#'   or `"wald"`.
#' @return Named list: `p_hat`, `ci_lower`, `ci_upper`.
#' @export
#' @examples
#' proportion_ci(35, 100)
proportion_ci <- function(successes, n, alpha = 0.05,
                           method = c("wilson", "exact", "wald")) {
  method <- match.arg(method)
  p <- successes / n
  z <- stats::qnorm(1 - alpha / 2)

  if (method == "wilson") {
    denom <- 1 + z^2 / n
    centre <- (p + z^2 / (2 * n)) / denom
    margin <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom
    ci <- c(centre - margin, centre + margin)
  } else if (method == "exact") {
    ci <- stats::qbeta(c(alpha / 2, 1 - alpha / 2),
                       c(successes, successes + 1),
                       c(n - successes + 1, n - successes))
  } else {
    margin <- z * sqrt(p * (1 - p) / n)
    ci <- c(p - margin, p + margin)
  }

  list(p_hat = p, ci_lower = pmax(0, ci[1]), ci_upper = pmin(1, ci[2]))
}

#' Odds ratio and 95% CI from a 2×2 contingency table
#'
#' @param table_2x2 A 2×2 matrix: rows are treatment, columns are outcome.
#' @param alpha Significance level.
#' @return Named list: `or`, `ci_lower`, `ci_upper`, `p_value`.
#' @export
odds_ratio_ci <- function(table_2x2, alpha = 0.05) {
  m <- as.matrix(table_2x2)
  result <- stats::fisher.test(m)
  list(
    or = as.numeric(result$estimate),
    ci_lower = result$conf.int[1],
    ci_upper = result$conf.int[2],
    p_value  = result$p.value
  )
}

#' Risk ratio (relative risk) with log-normal CI
#'
#' @param table_2x2 A 2×2 matrix: rows are exposure, columns are outcome (disease = col 1).
#' @param alpha Significance level.
#' @return Named list: `rr`, `ci_lower`, `ci_upper`.
#' @export
risk_ratio_ci <- function(table_2x2, alpha = 0.05) {
  m <- as.matrix(table_2x2)
  a <- m[1, 1]; b <- m[1, 2]; c <- m[2, 1]; d <- m[2, 2]
  n1 <- a + b; n2 <- c + d
  p1 <- a / n1; p2 <- c / n2
  rr <- p1 / p2
  log_se <- sqrt(1/a - 1/n1 + 1/c - 1/n2)
  z <- stats::qnorm(1 - alpha / 2)
  list(
    rr = rr,
    ci_lower = exp(log(rr) - z * log_se),
    ci_upper = exp(log(rr) + z * log_se)
  )
}

#' Risk difference (ARD) with Newcombe CI
#'
#' @param table_2x2 A 2×2 matrix: rows are exposure, columns are outcome.
#' @param alpha Significance level.
#' @return Named list: `rd`, `ci_lower`, `ci_upper`.
#' @export
risk_difference_ci <- function(table_2x2, alpha = 0.05) {
  m <- as.matrix(table_2x2)
  a <- m[1, 1]; b <- m[1, 2]; c <- m[2, 1]; d <- m[2, 2]
  n1 <- a + b; n2 <- c + d
  p1 <- a / n1; p2 <- c / n2
  rd <- p1 - p2
  z <- stats::qnorm(1 - alpha / 2)
  se <- sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
  list(
    rd = rd,
    ci_lower = rd - z * se,
    ci_upper = rd + z * se
  )
}


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

#' Cohen's d effect size
#'
#' @param x1 Numeric vector (group 1).
#' @param x2 Numeric vector (group 2).
#' @param pooled Use pooled SD (default `TRUE`). If `FALSE`, uses `sd(x2)`.
#' @return Numeric Cohen's d.
#' @export
cohens_d <- function(x1, x2, pooled = TRUE) {
  m1 <- mean(x1, na.rm = TRUE); m2 <- mean(x2, na.rm = TRUE)
  n1 <- sum(!is.na(x1)); n2 <- sum(!is.na(x2))
  s1 <- stats::sd(x1, na.rm = TRUE); s2 <- stats::sd(x2, na.rm = TRUE)
  sd_denom <- if (pooled) {
    sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2))
  } else s2
  (m1 - m2) / sd_denom
}

#' Hedges' g (bias-corrected Cohen's d)
#'
#' @inheritParams cohens_d
#' @return Numeric Hedges' g.
#' @export
hedges_g <- function(x1, x2) {
  d <- cohens_d(x1, x2, pooled = TRUE)
  n1 <- sum(!is.na(x1)); n2 <- sum(!is.na(x2))
  df <- n1 + n2 - 2
  correction <- 1 - 3 / (4 * df - 1)
  d * correction
}

#' Eta-squared from F-statistic
#'
#' @param f_stat F statistic.
#' @param df_between Degrees of freedom (numerator).
#' @param df_within Degrees of freedom (denominator).
#' @return Numeric eta-squared.
#' @export
eta_squared <- function(f_stat, df_between, df_within) {
  ss_between <- f_stat * df_between
  ss_total   <- ss_between + df_within
  ss_between / ss_total
}

#' Omega-squared (less biased than eta-squared)
#'
#' @inheritParams eta_squared
#' @param n Total sample size.
#' @return Numeric omega-squared.
#' @export
#' @examples
#' omega_squared(f_stat = 5.2, df_between = 2, df_within = 87, n = 90)
omega_squared <- function(f_stat, df_between, df_within, n) {
  (df_between * (f_stat - 1)) / (df_between * (f_stat - 1) + n)
}

#' Cramér's V for categorical association
#'
#' @param contingency_table A numeric matrix of observed counts.
#' @return Numeric Cramér's V in [0, 1].
#' @export
cramers_v <- function(contingency_table) {
  m <- as.matrix(contingency_table)
  result <- stats::chisq.test(m, correct = FALSE)
  chi2 <- as.numeric(result$statistic)
  n    <- sum(m)
  k    <- min(nrow(m), ncol(m))
  sqrt(chi2 / (n * (k - 1)))
}

#' Spearman rank correlation
#'
#' @param x Numeric vector.
#' @param y Numeric vector.
#' @return Named list: `rho`, `p_value`.
#' @export
spearman_rho <- function(x, y) {
  result <- stats::cor.test(x, y, method = "spearman", exact = FALSE)
  list(rho = as.numeric(result$estimate), p_value = result$p.value)
}

#' Kendall's tau-b
#'
#' @param x Numeric vector.
#' @param y Numeric vector.
#' @return Named list: `tau`, `p_value`.
#' @export
kendall_tau <- function(x, y) {
  result <- stats::cor.test(x, y, method = "kendall", exact = FALSE)
  list(tau = as.numeric(result$estimate), p_value = result$p.value)
}

#' Point-biserial correlation
#'
#' @param binary_var Binary numeric vector (0/1).
#' @param continuous_var Continuous numeric vector.
#' @return Named list: `r`, `p_value`.
#' @export
point_biserial_r <- function(binary_var, continuous_var) {
  result <- stats::cor.test(binary_var, continuous_var)
  list(r = as.numeric(result$estimate), p_value = result$p.value)
}


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

#' Power for a two-sample t-test
#'
#' Solve for any missing parameter (`n`, `delta`, `sd`, `sig.level`,
#' or `power`). Mirrors R's `power.t.test()`.
#'
#' @param n Sample size per group (NULL to solve for it).
#' @param delta Effect size (difference in means).
#' @param sd Standard deviation (pooled).
#' @param sig_level Type I error rate (alpha).
#' @param power Desired power (1 - beta).
#' @param alternative `"two.sided"` or `"one.sided"`.
#' @param type `"two.sample"`, `"one.sample"`, or `"paired"`.
#' @return Result of `stats::power.t.test()`.
#' @export
#' @examples
#' power_t_test(n = NULL, delta = 0.5, power = 0.80)
power_t_test <- function(n = NULL, delta = NULL, sd = 1,
                          sig_level = 0.05, power = NULL,
                          alternative = c("two.sided", "one.sided"),
                          type = c("two.sample", "one.sample", "paired")) {
  alternative <- match.arg(alternative)
  type <- match.arg(type)
  stats::power.t.test(
    n = n, delta = delta, sd = sd,
    sig.level = sig_level, power = power,
    alternative = alternative, type = type
  )
}

#' Power for a two-proportion z-test
#'
#' Mirrors R's `power.prop.test()`.
#'
#' @param n Sample size per group.
#' @param p1 Proportion in group 1.
#' @param p2 Proportion in group 2.
#' @param sig_level Type I error rate.
#' @param power Desired power.
#' @param alternative `"two.sided"` or `"one.sided"`.
#' @return Result of `stats::power.prop.test()`.
#' @export
#' @examples
#' power_prop_test(p1 = 0.30, p2 = 0.20, power = 0.80)
power_prop_test <- function(n = NULL, p1 = NULL, p2 = NULL,
                             sig_level = 0.05, power = NULL,
                             alternative = c("two.sided", "one.sided")) {
  alternative <- match.arg(alternative)
  stats::power.prop.test(
    n = n, p1 = p1, p2 = p2,
    sig.level = sig_level, power = power,
    alternative = alternative
  )
}

#' Sample size for logistic regression detecting a target odds ratio
#'
#' Uses the formula from Hsieh et al. (1998):
#' \deqn{n = \frac{(z_{\alpha/2} + z_\beta)^2}{p_1(1-p_1) [\log(OR)]^2}}
#'
#' @param p0 Prevalence under control.
#' @param or Target odds ratio.
#' @param alpha Significance level.
#' @param power Desired power.
#' @param two_sided Logical.
#' @return Integer sample size.
#' @export
#' @references
#'   Hsieh FY, Bloch DA, Larsen MD (1998). A simple method of sample size
#'   calculation for linear and logistic regression.
#'   *Statistics in Medicine*, 17(14):1623–1634.
sample_size_logistic <- function(p0, or, alpha = 0.05, power = 0.80,
                                  two_sided = TRUE) {
  p1 <- (or * p0) / (1 - p0 + or * p0)
  z_a <- stats::qnorm(if (two_sided) 1 - alpha / 2 else 1 - alpha)
  z_b <- stats::qnorm(power)
  p_bar <- (p0 + p1) / 2
  n <- as.integer(ceiling((z_a + z_b)^2 / (p_bar * (1 - p_bar) * (log(or))^2)))
  n
}
