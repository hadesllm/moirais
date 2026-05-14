# SPDX-License-Identifier: MIT OR Apache-2.0

#' Van der Waerden two-sample normal-scores location test
#' (Gibbons Ch 8.3.2)
#'
#' Scores a_i = qnorm(R_i / (N + 1)); statistic = sum over the
#' first sample.
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
van_der_waerden_test <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  m <- length(x); n <- length(y); N <- m + n
  if (m < 2 || n < 2) {
    return(list(statistic = NA_real_, p_value = NA_real_, z = NA_real_,
                n = N, m = m,
                method = "Van der Waerden normal-scores test"))
  }
  pooled <- c(x, y)
  ranks <- rank(pooled)
  s <- stats::qnorm(ranks / (N + 1))
  T <- sum(s[1:m])
  Var_T <- (m * n / (N * (N - 1))) * sum(s^2)
  z <- T / sqrt(Var_T)
  p <- 2 * (1 - stats::pnorm(abs(z)))
  list(
    statistic = T,
    p_value = p,
    z = z,
    n = N,
    m = m,
    method = "Van der Waerden normal-scores test"
  )
}
