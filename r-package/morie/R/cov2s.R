# SPDX-License-Identifier: MIT OR Apache-2.0

#' Two-sample placement coverage (Gibbons Ch 2.11.2)
#'
#' Block frequencies of Y in the m+1 intervals defined by the
#' ordered X-sample.  Under H0 the expected block proportion is
#' `1 / (m + 1)`.
#'
#' @param x Numeric vector (first sample).
#' @param y Numeric vector (second sample).
#' @return Named list: block_freq, block_prop, expected_prop, m, n,
#'   cumulative, method.
#' @export
two_sample_coverage <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  m <- length(x); n <- length(y)
  if (m < 1 || n < 1) {
    return(list(block_freq = integer(0), block_prop = numeric(0),
                expected_prop = NA_real_, m = m, n = n,
                cumulative = 0L,
                method = "Two-sample coverage probability"))
  }
  xs <- sort(x)
  # findInterval(y, xs) gives 0..m where 0 means y <= xs[1]-eps and m means y > xs[m]
  idx <- findInterval(y, xs, left.open = TRUE) # 0..m
  block_freq <- tabulate(idx + 1L, nbins = m + 1L)
  block_prop <- block_freq / n
  list(
    block_freq = as.integer(block_freq),
    block_prop = block_prop,
    expected_prop = 1 / (m + 1),
    m = m,
    n = n,
    cumulative = sum(block_freq),
    method = "Two-sample coverage probability"
  )
}
