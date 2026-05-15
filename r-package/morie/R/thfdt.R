# SPDX-License-Identifier: AGPL-3.0-or-later

#' Terry-Hoeffding (Fisher-Yates) two-sample normal-scores test
#' (Gibbons Ch 8.3.1)
#'
#' Replaces pooled ranks with Blom-approximated normal scores
#' a_i = qnorm((R_i - 3/8) / (N + 1/4)).  Statistic T = sum of
#' scores from the first sample.
#'
#' @param x,y Numeric vectors.
#' @return Named list: statistic, p_value, z, n, m.
#' @importFrom stats qnorm pnorm
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
terry_hoeffding_test <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  m <- length(x); n <- length(y); N <- m + n
  if (m < 2 || n < 2) {
    return(list(statistic = NA_real_, p_value = NA_real_, z = NA_real_,
                n = N, m = m,
                method = "Terry-Hoeffding (Fisher-Yates) normal-scores test"))
  }
  pooled <- c(x, y)
  ranks <- rank(pooled)
  a <- stats::qnorm((ranks - 3/8) / (N + 1/4))
  T <- sum(a[1:m])
  sum_a2 <- sum(a^2)
  Var_T <- (m * n / (N * (N - 1))) * sum_a2
  z <- T / sqrt(Var_T)
  p <- 2 * (1 - stats::pnorm(abs(z)))
  list(
    statistic = T,
    p_value = p,
    z = z,
    n = N,
    m = m,
    method = "Terry-Hoeffding (Fisher-Yates) normal-scores test"
  )
}
