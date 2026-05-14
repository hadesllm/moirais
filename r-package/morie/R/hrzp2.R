# SPDX-License-Identifier: MIT OR Apache-2.0

#' Silverman bandwidth selector for PLR
#'
#' @param x Numeric vector.
#' @param y Numeric vector (unused; kept for API parity).
#' @param c Silverman multiplier (default 1.06).
#' @return Named list with estimate (h), n, sigma, c, method.
#' @keywords internal
hrzp2 <- function(x, y, c = 1.06) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5) return(list(estimate = NA_real_, n = n,
                          method = "plr-bandwidth (insufficient data)"))
  s <- stats::sd(x)
  iqr <- diff(stats::quantile(x, c(0.25, 0.75), na.rm = TRUE))
  sigma <- if (iqr > 0) min(s, iqr / 1.349) else s
  if (sigma <= 0) sigma <- max(s, 1e-6)
  h <- as.numeric(c * sigma * n ^ (-1/5))
  list(estimate = h, n = n, sigma = as.numeric(sigma), c = c,
       method = "Silverman h = c * sigma * n^(-1/5)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzp2
#' @keywords internal
#' @export
horowitz_plr_bandwidth <- hrzp2
