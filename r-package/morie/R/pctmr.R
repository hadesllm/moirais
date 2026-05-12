# SPDX-License-Identifier: GPL-2.0-only

#' Percentile-modified rank (Gastwirth) two-sample test
#' (Gibbons Ch 8.3.3)
#'
#' Trims central ranks; only tail ranks contribute.  Score:
#'   a_i = max(R_i - (1-q)(N+1), 0) - max(q(N+1) - R_i, 0)
#'
#' @param x,y Numeric vectors.
#' @param q Tail fraction in (0, 0.5).  Default 0.25.
#' @return Named list: statistic, p_value, z, n, m, q.
#' @importFrom stats pnorm
#' @export
percentile_modified_rank <- function(x, y, q = 0.25) {
  x <- as.numeric(x); y <- as.numeric(y)
  m <- length(x); n <- length(y); N <- m + n
  if (m < 2 || n < 2) {
    return(list(statistic = NA_real_, p_value = NA_real_, z = NA_real_,
                n = N, m = m, q = q,
                method = "Percentile-modified rank test"))
  }
  if (!(q > 0 && q < 0.5))
    stop("q must lie strictly between 0 and 0.5")
  pooled <- c(x, y)
  R <- rank(pooled)
  upper_cut <- (1 - q) * (N + 1)
  lower_cut <- q * (N + 1)
  a <- pmax(R - upper_cut, 0) - pmax(lower_cut - R, 0)
  T <- sum(a[1:m])
  Var_T <- (m * n / (N * (N - 1))) * sum(a^2)
  z <- T / sqrt(Var_T)
  p <- 2 * (1 - stats::pnorm(abs(z)))
  list(
    statistic = T,
    p_value = p,
    z = z,
    n = N,
    m = m,
    q = q,
    method = "Percentile-modified rank (Gastwirth) test"
  )
}
