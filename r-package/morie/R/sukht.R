# SPDX-License-Identifier: AGPL-3.0-or-later

#' Sukhatme two-sample scale test (Gibbons Ch 9.7)
#'
#' Mann-Whitney U on the absolute deviations from the pooled median.
#' Tests equality of scales given (approximately) equal medians.
#'
#' @param x,y Numeric vectors.
#' @return Named list: statistic (z), p_value, U, n, m.
#' @importFrom stats wilcox.test median
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
sukhatme_test <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  m <- length(x); n <- length(y); N <- m + n
  if (m < 2 || n < 2) {
    return(list(statistic = NA_real_, p_value = NA_real_, U = NA_real_,
                n = N, m = m,
                method = "Sukhatme scale test"))
  }
  pooled_med <- stats::median(c(x, y))
  ax <- abs(x - pooled_med)
  ay <- abs(y - pooled_med)
  tst <- suppressWarnings(stats::wilcox.test(ax, ay, exact = FALSE,
                                              correct = FALSE))
  U <- as.numeric(tst$statistic)
  E_U <- m * n / 2
  Var_U <- m * n * (N + 1) / 12
  z <- (U - E_U) / sqrt(Var_U)
  list(
    statistic = z,
    p_value = as.numeric(tst$p.value),
    U = U,
    n = N,
    m = m,
    method = "Sukhatme scale test (Mann-Whitney on |.-median|)"
  )
}
