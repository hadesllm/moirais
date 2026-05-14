# SPDX-License-Identifier: MIT OR Apache-2.0

#' Mathematical-statistics / simulation / computation toolkit (R parity)
#'
#' R parity of \code{morie.mrm_mathstats}.  Closes the Chapter-2
#' coverage gap from designexptr.org/mathematical-statistics-
#' simulation-and-computation.html.
#'
#' @references
#' Wilks, S. S. (1962). Mathematical Statistics. Wiley.
#' Casella, G. & Berger, R. L. (2002). Statistical Inference. Duxbury.
#' Lehmann, E. L. & Romano, J. P. (2005). Testing Statistical Hypotheses.
#'
#' @name mrm_mathstats
NULL


#' One-proportion test (binomial exact + Wald approximation)
#'
#' @param x Number of successes.
#' @param n Number of trials.
#' @param p0 Null-hypothesis proportion.
#' @param alpha CI level (default 0.05 -> 95% CI).
#' @return Named list with p_hat, p0, n, z_wald, p_value_wald,
#'   p_value_exact, ci95_wald_lower/upper, ci95_exact_lower/upper,
#'   interpretation.
#' @examples
#' # H0: proportion = 0.5 against the observed 58/100 successes
#' mrm_oneprop_test(x = 58, n = 100, p0 = 0.5)
#' @export
mrm_oneprop_test <- function(x, n, p0, alpha = 0.05) {
  if (n <= 0 || x < 0 || x > n) stop("invalid x, n")
  p_hat <- x / n
  se_null <- sqrt(p0 * (1 - p0) / n)
  z <- if (se_null > 0) (p_hat - p0) / se_null else NA_real_
  p_wald <- 2 * (1 - stats::pnorm(abs(z)))
  bt <- stats::binom.test(x, n, p = p0)
  p_exact <- bt$p.value
  se <- if (p_hat > 0 && p_hat < 1) sqrt(p_hat * (1 - p_hat) / n) else 0
  z_a <- stats::qnorm(1 - alpha / 2)
  cp <- stats::binom.test(x, n)$conf.int
  list(
    p_hat = round(p_hat, 6),
    p0 = round(p0, 6), n = as.integer(n),
    z_wald = round(z, 4),
    p_value_wald = p_wald, p_value_exact = p_exact,
    ci95_wald_lower = round(max(0, p_hat - z_a * se), 6),
    ci95_wald_upper = round(min(1, p_hat + z_a * se), 6),
    ci95_exact_lower = round(cp[1], 6),
    ci95_exact_upper = round(cp[2], 6),
    interpretation = sprintf(
      "p_hat = %.4f, H0: p = %g; exact p = %.3g (%s H0 at alpha=%g).",
      p_hat, p0, p_exact,
      if (p_exact < alpha) "reject" else "fail to reject",
      alpha
    )
  )
}


#' Two-proportion test (chi-square + Fisher exact + Wald)
#'
#' @param x1,n1 Successes and trials in group 1.
#' @param x2,n2 Successes and trials in group 2.
#' @param alpha CI level (default 0.05).
#' @return Named list with p1, p2, diff, chi2, df, p_value_chi2,
#'   p_value_fisher, z_wald, p_value_wald, ci95_diff_lower/upper,
#'   interpretation.
#' @examples
#' # Compare 47/100 vs 31/100; two-sided test.
#' mrm_twoprop_test(x1 = 47, n1 = 100, x2 = 31, n2 = 100)
#' @export
mrm_twoprop_test <- function(x1, n1, x2, n2, alpha = 0.05) {
  if (n1 <= 0 || n2 <= 0 || x1 < 0 || x2 < 0) {
    stop("invalid sample sizes / counts")
  }
  p1 <- x1 / n1; p2 <- x2 / n2
  tbl <- matrix(c(x1, n1 - x1, x2, n2 - x2), nrow = 2, byrow = TRUE)
  ch <- suppressWarnings(stats::chisq.test(tbl, correct = FALSE))
  fi <- stats::fisher.test(tbl, alternative = "two.sided")
  se <- sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
  z_w <- if (se > 0) (p1 - p2) / se else NA_real_
  p_wald <- 2 * (1 - stats::pnorm(abs(z_w)))
  z_a <- stats::qnorm(1 - alpha / 2)
  diff <- p1 - p2
  list(
    p1 = round(p1, 6), p2 = round(p2, 6),
    diff = round(diff, 6),
    chi2 = round(as.numeric(ch$statistic), 4),
    df   = as.integer(ch$parameter),
    p_value_chi2 = as.numeric(ch$p.value),
    p_value_fisher = as.numeric(fi$p.value),
    z_wald = round(z_w, 4),
    p_value_wald = p_wald,
    ci95_diff_lower = round(diff - z_a * se, 6),
    ci95_diff_upper = round(diff + z_a * se, 6),
    interpretation = sprintf(
      "p1=%.4f, p2=%.4f; Delta=%.4f; chi2(%d)=%.3f, p_chi2=%.3g, p_Fisher=%.3g.",
      p1, p2, diff, as.integer(ch$parameter), as.numeric(ch$statistic),
      as.numeric(ch$p.value), as.numeric(fi$p.value)
    )
  )
}


#' Chi-square test for variance (Wilks 1962)
#'
#' @param sample Numeric vector (assumed iid normal).
#' @param sigma0_sq Null hypothesis variance.
#' @param alpha CI level (default 0.05).
#' @return Named list with s_sq, sigma0_sq, chi2_stat, df,
#'   p_value_two_sided, p_value_one_sided_greater/less,
#'   ci95_lower/upper, interpretation.
#' @examples
#' set.seed(2026)
#' x <- rnorm(50, mean = 0, sd = 1.2)
#' # H0: variance = 1.
#' mrm_var_test(sample = x, sigma0_sq = 1)
#' @export
mrm_var_test <- function(sample, sigma0_sq, alpha = 0.05) {
  x <- as.numeric(sample); x <- x[is.finite(x)]
  n <- length(x)
  if (n < 2L) stop("need >= 2 observations")
  s_sq <- stats::var(x)
  df_ <- n - 1L
  stat <- df_ * s_sq / sigma0_sq
  p_lower <- stats::pchisq(stat, df_)
  p_upper <- 1 - p_lower
  p_two <- 2 * min(p_lower, p_upper)
  lo <- df_ * s_sq / stats::qchisq(1 - alpha / 2, df_)
  hi <- df_ * s_sq / stats::qchisq(alpha / 2, df_)
  list(
    s_sq = round(s_sq, 6),
    sigma0_sq = round(sigma0_sq, 6),
    chi2_stat = round(stat, 4), df = df_,
    p_value_two_sided = p_two,
    p_value_one_sided_greater = p_upper,
    p_value_one_sided_less = p_lower,
    ci95_lower = round(lo, 6),
    ci95_upper = round(hi, 6),
    interpretation = sprintf(
      "s^2 = %.4f; H0: sigma^2 = %g; chi2(%d) = %.3f, two-sided p = %.3g.",
      s_sq, sigma0_sq, df_, stat, p_two
    )
  )
}


#' Q-Q plot coordinates against a reference distribution
#'
#' @param sample Numeric vector.
#' @param dist Either \code{"norm"}, \code{"exp"}, \code{"unif"},
#'   \code{"t"}, \code{"chisq"} (any base \code{q<dist>} works).
#' @param ... Additional parameters passed to \code{q<dist>}.
#' @return data.frame with rank, empirical, theoretical,
#'   plotting_position columns (Blom 1958 plotting positions).
#' @examples
#' set.seed(2026)
#' x <- rnorm(100)
#' qq <- mrm_qq_plot(x, dist = "norm")
#' head(qq)
#' # plot(qq$theoretical, qq$empirical); abline(0, 1)
#' @export
mrm_qq_plot <- function(sample, dist = "norm", ...) {
  x <- sort(as.numeric(sample)); x <- x[is.finite(x)]
  n <- length(x)
  if (n < 2L) stop("need >= 2 observations")
  p <- ((seq_len(n)) - 0.375) / (n + 0.25)
  qfun <- get(paste0("q", dist), envir = asNamespace("stats"))
  theoretical <- qfun(p, ...)
  data.frame(
    rank = seq_len(n),
    empirical = x,
    theoretical = theoretical,
    plotting_position = p
  )
}


#' Central Limit Theorem demonstrator
#'
#' Generate sample means from a base distribution.
#'
#' @param base_distribution Distribution suffix passed to
#'   \code{r<dist>} (e.g. \code{"unif"}, \code{"exp"}, \code{"pois"}).
#' @param n_samples Number of sample means.
#' @param sample_size Size of each sample.
#' @param seed RNG seed.
#' @param ... Additional parameters passed to \code{r<dist>}.
#' @return data.frame with sample_index, sample_mean, z_score.
#' @examples
#' # 1000 sample means of size 30 from an exponential(1) base;
#' # standardised z-scores converge to N(0,1):
#' res <- mrm_clt_demo(base_distribution = "exp",
#'                     n_samples = 1000L,
#'                     sample_size = 30L,
#'                     seed = 42L, rate = 1)
#' summary(res$z_score)
#' # mean ~ 0, sd ~ 1
#' @export
mrm_clt_demo <- function(base_distribution = "unif",
                          n_samples = 1000L,
                          sample_size = 30L,
                          seed = 42L, ...) {
  set.seed(seed)
  rfun <- get(paste0("r", base_distribution), envir = asNamespace("stats"))
  means <- vapply(seq_len(n_samples),
                   function(i) mean(rfun(sample_size, ...)),
                   numeric(1))
  data.frame(
    sample_index = seq_len(n_samples),
    sample_mean = means,
    z_score = (means - mean(means)) / stats::sd(means)
  )
}


#' Probability Integral Transform (PIT)
#'
#' If X ~ F, then F(X) ~ Uniform(0,1).  Returned U should be approx
#' uniform if the assumed F is correct.  Attaches a KS p-value of U
#' against Uniform(0,1) as the diagnostic for fit quality.
#'
#' @param sample Numeric vector.
#' @param dist Distribution suffix for \code{p<dist>}.
#' @param ... Additional parameters for \code{p<dist>}.
#' @return data.frame with raw, U columns and attributes ks_stat,
#'   ks_pvalue.
#' @examples
#' set.seed(2026)
#' x <- rnorm(200)
#' # Under correct distributional assumption, U should be ~Uniform(0,1):
#' pit <- mrm_pit(x, dist = "norm")
#' attr(pit, "ks_pvalue")  # large p-value => no evidence against fit
#' # If we deliberately misspecify (claim t_3 fits the normal sample):
#' pit_wrong <- mrm_pit(x, dist = "t", df = 3)
#' attr(pit_wrong, "ks_pvalue")  # small p-value => misspecification detected
#' @export
mrm_pit <- function(sample, dist = "norm", ...) {
  x <- as.numeric(sample); x <- x[is.finite(x)]
  pfun <- get(paste0("p", dist), envir = asNamespace("stats"))
  U <- pfun(x, ...)
  ks <- suppressWarnings(stats::ks.test(U, "punif"))
  out <- data.frame(raw = x, U = U)
  attr(out, "ks_stat") <- as.numeric(ks$statistic)
  attr(out, "ks_pvalue") <- as.numeric(ks$p.value)
  out
}
