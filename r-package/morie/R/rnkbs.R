# SPDX-License-Identifier: AGPL-3.0-or-later

#' Mann's rank test for randomness (Gibbons Ch 3.5)
#'
#' Kendall tau between the observation and its time index t = 1..n.
#' Tests H0: no monotone trend.
#'
#' @param x Numeric vector of sequential observations.
#' @return Named list: statistic (tau), p_value, n, inversions, z.
#' @importFrom stats cor.test
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
rank_based_test <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  if (n < 3) {
    return(list(statistic = NA_real_, p_value = NA_real_, n = n,
                inversions = 0L, z = NA_real_,
                method = "Mann's rank test for randomness"))
  }
  t <- seq_len(n)
  ct <- suppressWarnings(stats::cor.test(t, x, method = "kendall"))
  # Count inversions
  inv <- 0L
  for (i in seq_len(n - 1)) {
    inv <- inv + sum(x[(i + 1):n] < x[i])
  }
  tau <- as.numeric(ct$estimate)
  z <- tau * sqrt(9 * n * (n - 1) / (2 * (2 * n + 5)))
  list(
    statistic = tau,
    p_value = as.numeric(ct$p.value),
    n = n,
    inversions = inv,
    z = z,
    method = "Mann's rank test for randomness (Kendall tau vs time)"
  )
}
