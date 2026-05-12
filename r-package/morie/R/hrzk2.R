# SPDX-License-Identifier: GPL-2.0-only

#' Nadaraya-Watson kernel regression
#'
#' @param x Numeric covariate vector.
#' @param y Numeric response vector.
#' @param bandwidth Optional kernel bandwidth (Silverman default).
#' @param grid Optional evaluation grid (defaults to \code{x}).
#' @return Named list with estimate, se, bandwidth, n, method.
#' @keywords internal
hrzk2 <- function(x, y, bandwidth = NULL, grid = NULL) {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (n < 2 || length(y) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "NW (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(x) else as.numeric(bandwidth)
  if (h <= 0) h <- .hrz_silverman(x)
  g <- if (is.null(grid)) x else as.numeric(grid)
  u <- outer(g, x, `-`) / h
  w <- exp(-0.5 * u^2)
  wsum <- rowSums(w); safe <- ifelse(wsum > 0, wsum, 1)
  m_hat <- (w %*% y) / safe
  resid <- outer(rep(1, length(g)), y) - matrix(m_hat, length(g), n)
  sigma2 <- rowSums(w * resid^2) / safe
  f_hat <- wsum / (n * h * sqrt(2 * pi))
  se <- sqrt(pmax(sigma2, 0) * .hrz_R_K_gaussian / (n * h * pmax(f_hat, 1e-12)))
  list(estimate = as.numeric(m_hat), se = as.numeric(se),
       bandwidth = h, n = n,
       method = "Nadaraya-Watson kernel regression (Gaussian)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzk2
#' @keywords internal
#' @export
horowitz_kernel_regression <- hrzk2
