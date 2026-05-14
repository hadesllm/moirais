# SPDX-License-Identifier: MIT OR Apache-2.0

#' One-sample coverage probability (Gibbons Ch 2.11.1)
#'
#' For an ordered sample the coverages U_i = F(X_(i)) - F(X_(i-1))
#' are i.i.d. Beta(1, n) under H0.  Returns empirical coverages
#' (rank-based) plus the cumulative coverage F(X_(n)) - F(X_(1)).
#'
#' @param x Numeric vector.
#' @return Named list: coverages, cumulative, expected, n,
#'   sample_min, sample_max, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
one_sample_coverage <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  if (n < 2) {
    return(list(coverages = numeric(0), cumulative = NA_real_,
                expected = NA_real_, n = n,
                method = "One-sample coverage probability"))
  }
  xs <- sort(x)
  ranks <- seq_len(n) / (n + 1)
  coverages <- diff(c(0, ranks, 1))
  list(
    coverages = coverages,
    cumulative = ranks[n] - ranks[1],
    expected = 1 / (n + 1),
    n = n,
    sample_min = xs[1],
    sample_max = xs[n],
    method = "One-sample coverage probability"
  )
}
