# SPDX-License-Identifier: AGPL-3.0-or-later

#' Local-linear regression estimator
#'
#' @param x Numeric covariate vector.
#' @param y Numeric response vector.
#' @param bandwidth Optional kernel bandwidth (Silverman default).
#' @param grid Optional evaluation grid (defaults to \code{x}).
#' @return Named list with estimate, se, bandwidth, n, method.
#' @keywords internal
hrzk3 <- function(x, y, bandwidth = NULL, grid = NULL) {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (n < 3 || length(y) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "local-linear (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(x) else as.numeric(bandwidth)
  if (h <= 0) h <- .hrz_silverman(x)
  g <- if (is.null(grid)) x else as.numeric(grid)
  m_hat <- numeric(length(g)); se <- numeric(length(g))
  for (i in seq_along(g)) {
    u <- (x - g[i]) / h
    w <- exp(-0.5 * u^2)
    if (sum(w) <= 1e-12) { m_hat[i] <- NA; se[i] <- NA; next }
    X <- cbind(1, x - g[i])
    WX <- X * w
    XtWX <- t(X) %*% WX
    beta <- tryCatch(solve(XtWX, t(WX) %*% y),
                     error = function(e) MASS::ginv(XtWX) %*% (t(WX) %*% y))
    m_hat[i] <- beta[1]
    r <- y - X %*% beta
    sigma2 <- sum(w * r^2) / max(sum(w), 1e-12)
    f_hat <- sum(w) / (n * h * sqrt(2 * pi))
    se[i] <- sqrt(max(sigma2, 0) * .hrz_R_K_gaussian / (n * h * max(f_hat, 1e-12)))
  }
  list(estimate = if (length(m_hat) == 1) m_hat[1] else m_hat,
       se = if (length(se) == 1) se[1] else se,
       bandwidth = h, n = n,
       method = "Local-linear regression (Gaussian kernel)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzk3
#' @keywords internal
#' @export
horowitz_local_linear <- hrzk3
