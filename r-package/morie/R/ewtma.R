# SPDX-License-Identifier: AGPL-3.0-or-later

#' EWMA volatility (RiskMetrics 1996)
#'
#' @inheritParams garch_fit
#' @param lambda Decay factor in (0,1). Default 0.94 (daily RiskMetrics).
#' @return Named list with \code{conditional_variance, conditional_volatility,
#'   lambda, n, last_variance, last_volatility, method}.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ewma_volatility <- function(x, lambda = 0.94) {
  r <- as.numeric(x); n <- length(r)
  if (n < 2) stop("Need >=2 obs.")
  if (lambda <= 0 || lambda >= 1) stop("lambda must be in (0,1).")
  r2 <- r^2
  s2 <- numeric(n); s2[1] <- r2[1]
  for (t in 2:n) s2[t] <- lambda * s2[t - 1] + (1 - lambda) * r2[t - 1]
  list(conditional_variance = s2,
       conditional_volatility = sqrt(s2),
       lambda = lambda, n = n,
       last_variance = s2[n],
       last_volatility = sqrt(s2[n]),
       method = "EWMA RiskMetrics")
}
