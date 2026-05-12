# SPDX-License-Identifier: GPL-2.0-only

#' Posterior mean / variance of G(A) for DP(alpha, G0) and A = (A_lower, A_upper].
#'
#' @param x Numeric data vector.
#' @param alpha DP concentration parameter (default 1).
#' @param A_lower Optional numeric lower bound of set A (default -Inf).
#' @param A_upper Optional numeric upper bound of set A (default mean(x)).
#' @param base_mean Numeric base-measure mean (default 0).
#' @param base_sd Numeric base-measure sd (default 1).
#' @return Named list with estimate, se, prior_mean, prior_var, n_A, n, alpha, method.
#' @export
ghosal_moment_matching <- function(x, alpha = 1.0, A_lower = NULL,
                                     A_upper = NULL, base_mean = 0,
                                     base_sd = 1) {
  x <- as.numeric(x); n <- length(x)
  if (is.null(A_lower)) A_lower <- -Inf
  if (is.null(A_upper)) A_upper <- if (n) mean(x) else 0
  G0_A <- max(0, min(1, stats::pnorm(A_upper, base_mean, base_sd)
                         - stats::pnorm(A_lower, base_mean, base_sd)))
  prior_mean <- G0_A
  prior_var <- G0_A * (1 - G0_A) / (alpha + 1)
  n_A <- if (n) sum(x > A_lower & x <= A_upper) else 0L
  post_mean <- (alpha * G0_A + n_A) / (alpha + n)
  post_var  <- post_mean * (1 - post_mean) / (alpha + n + 1)
  list(estimate = post_mean, se = sqrt(max(post_var, 0)),
       prior_mean = prior_mean, prior_var = prior_var,
       n_A = as.integer(n_A), n = n, alpha = alpha,
       method = "DP moment-matching (Ferguson 1973)")
}
