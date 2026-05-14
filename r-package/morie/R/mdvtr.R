# SPDX-License-Identifier: MIT OR Apache-2.0

#' Median voter theorem (Armstrong et al. Ch 2)
#'
#' Black (1948) median-voter theorem: with single-peaked preferences in
#' 1-D, the Condorcet winner equals the median ideal point. Laplace
#' asymptotic SE 1.2533 * s/sqrt(n) for normal-like data.
#'
#' @param x Numeric vector of voter ideal points.
#' @return Named list with `estimate`, `se`, `ci_lower`, `ci_upper`,
#'   `n`, `method`.
#' @references Armstrong et al. (2014), Ch 2.
#' @export
mdvtr <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  n <- length(x)
  if (n == 0L) {
    return(list(estimate = NA_real_, se = NA_real_, ci_lower = NA_real_,
                ci_upper = NA_real_, n = 0L, method = "median_voter"))
  }
  est <- stats::median(x)
  se  <- if (n > 1L) 1.2533141373 * stats::sd(x) / sqrt(n) else NA_real_
  ci_lo <- if (is.finite(se)) est - 1.96 * se else NA_real_
  ci_hi <- if (is.finite(se)) est + 1.96 * se else NA_real_
  list(estimate = est, se = se, ci_lower = ci_lo, ci_upper = ci_hi,
       n = n, method = "Median voter theorem")
}

#' @keywords internal
#' @rdname mdvtr
#' @export
median_voter <- mdvtr
