# SPDX-License-Identifier: GPL-2.0-only

#' Empirical process indexed by a function class
#'
#' G_n(f) = sqrt(n) * (P_n - P)(f).  Returns the standardised CLT
#' statistic sqrt(n) * (P_n(f) - mu0) plus the empirical L2(P_n)
#' standard deviation sigma_n(f).
#'
#' @param x Numeric vector of IID observations.
#' @param f Optional function applied element-wise to \code{x}.
#' @param mu0 Hypothesised P(f) under H_0 (default 0).
#' @return Named list: estimate, se, n, method.
#' @references Kosorok (2008), Ch 2.
#' @export
ksr01_kosorok_empirical_process <- function(x, f = NULL, mu0 = 0) {
  x  <- as.numeric(x)
  n  <- length(x)
  fx <- if (is.null(f)) x else vapply(x, f, numeric(1))
  pn <- mean(fx)
  estimate <- sqrt(n) * (pn - mu0)
  se <- if (n > 1L) stats::sd(fx) else NA_real_
  list(
    estimate = estimate,
    se       = se,
    n        = n,
    method   = "Empirical process G_n(f) = sqrt(n)(P_n - P)(f)"
  )
}

# CANONICAL TEST
# set.seed(0); xs <- rnorm(200); r <- ksr01_kosorok_empirical_process(xs); r

#' @rdname ksr01_kosorok_empirical_process
#' @keywords internal
#' @export
kosorok_empirical_process <- ksr01_kosorok_empirical_process
