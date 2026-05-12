# SPDX-License-Identifier: GPL-2.0-only

#' Power function of the two-sided sign test (Gibbons Ch 5.4.4)
#'
#' Builds the discrete rejection region under H0: p = 0.5 with size
#' <= alpha, then evaluates power at the alternative p_alt.
#'
#' @param x Numeric vector (only `length(x != mu0)` is used).
#' @param mu0 Null median.
#' @param p_alt Alternative success probability P(X > mu0).
#' @param alpha Test level.
#' @return Named list: statistic (power), n, p_alt, alpha, size,
#'   k_lower, k_upper.
#' @importFrom stats dbinom
#' @export
sign_test_power <- function(x, mu0 = 0, p_alt = 0.7, alpha = 0.05) {
  x <- as.numeric(x)
  n <- sum(x != mu0)
  if (n < 1 || !(p_alt > 0 && p_alt < 1)) {
    return(list(statistic = NA_real_, n = n, p_alt = p_alt,
                alpha = alpha,
                method = "Sign-test power"))
  }
  k_grid <- 0:n
  null_pmf <- stats::dbinom(k_grid, n, 0.5)
  ord <- order(null_pmf)
  cum <- 0; reject <- logical(n + 1)
  for (k in ord) {
    if (cum + null_pmf[k + 1] <= alpha) {
      reject[k + 1] <- TRUE
      cum <- cum + null_pmf[k + 1]
    } else break
  }
  size <- cum
  if (!any(reject)) {
    return(list(statistic = 0, n = n, p_alt = p_alt,
                alpha = alpha, size = 0,
                method = "Sign-test power",
                warnings = sprintf("No rejection region of size <= %g for n=%d",
                                    alpha, n)))
  }
  alt_pmf <- stats::dbinom(k_grid, n, p_alt)
  power <- sum(alt_pmf[reject])
  rej_ks <- k_grid[reject]
  list(
    statistic = power,
    n = n,
    p_alt = p_alt,
    alpha = alpha,
    size = size,
    k_lower = min(rej_ks),
    k_upper = max(rej_ks),
    method = "Two-sided sign-test power function"
  )
}
