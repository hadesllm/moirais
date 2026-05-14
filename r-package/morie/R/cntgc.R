# SPDX-License-Identifier: MIT OR Apache-2.0

#' Pearson contingency coefficient C (Gibbons Ch 14.2.1)
#'
#' C = sqrt(chi^2 / (chi^2 + n)).  Also reports Cramer's V and the
#' maximum attainable C = sqrt((min(r,c)-1)/min(r,c)).
#'
#' @param x A 2-D contingency table of counts.
#' @return Named list: statistic (C), cramers_v, chi2, p_value, df,
#'   max_C, n.
#' @importFrom stats chisq.test
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
contingency_coefficient <- function(x) {
  X <- as.matrix(x)
  if (length(dim(X)) != 2L || length(X) == 0L) {
    return(list(statistic = NA_real_, cramers_v = NA_real_,
                chi2 = NA_real_, p_value = NA_real_, df = NA_integer_,
                max_C = NA_real_, n = 0L,
                method = "Pearson contingency coefficient"))
  }
  ct <- suppressWarnings(stats::chisq.test(X, correct = FALSE))
  n_total <- sum(X)
  chi2 <- as.numeric(ct$statistic)
  C <- sqrt(chi2 / (chi2 + n_total))
  r <- nrow(X); c <- ncol(X); mn <- min(r, c)
  V <- if (mn > 1) sqrt(chi2 / (n_total * (mn - 1))) else NA_real_
  max_C <- if (mn > 1) sqrt((mn - 1) / mn) else NA_real_
  list(
    statistic = C,
    cramers_v = V,
    chi2 = chi2,
    p_value = as.numeric(ct$p.value),
    df = as.integer(ct$parameter),
    max_C = max_C,
    n = as.integer(n_total),
    method = "Pearson contingency coefficient"
  )
}
