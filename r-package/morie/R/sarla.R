# SPDX-License-Identifier: GPL-2.0-only
#' Spatial autoregressive lag model (SAR lag, ML).
#'
#' Y = rho W Y + X beta + eps,  eps ~ N(0, sigma2 I).
#' Concentrated log-likelihood in rho.
#'
#' @param x Design matrix (n by p).
#' @param y Response, length n.
#' @param w n-by-n weights matrix.
#' @return Named list: estimate, se, rho, sigma2, n, method.
#' @references Anselin (1988); Schabenberger & Gotway (2005), Ch 7.
#' @export
sarla <- function(x, y, w) {
  X <- as.matrix(x); y <- as.numeric(y); W <- as.matrix(w)
  n <- length(y); p <- ncol(X)
  if (nrow(X) != n || any(dim(W) != c(n, n)))
    stop("shape mismatch among x, y, w")
  I <- diag(n)
  XtX_inv <- solve(crossprod(X))
  M <- I - X %*% XtX_inv %*% t(X)
  e0 <- M %*% y
  e1 <- M %*% (W %*% y)
  neg_ll <- function(rho) {
    e <- e0 - rho * e1
    sigma2 <- as.numeric(sum(e ^ 2)) / n
    A <- I - rho * W
    det_sign <- determinant(A, logarithm = TRUE)
    if (det_sign$sign <= 0 || sigma2 <= 0) return(1e12)
    logdetA <- as.numeric(det_sign$modulus)
    0.5 * n * log(2 * pi * sigma2) - logdetA + 0.5 * n
  }
  res <- stats::optimize(neg_ll, interval = c(-0.99, 0.99))
  rho <- res$minimum
  Wy <- W %*% y
  y_star <- as.numeric(y - rho * Wy)
  beta <- as.numeric(XtX_inv %*% crossprod(X, y_star))
  e <- y_star - as.numeric(X %*% beta)
  sigma2 <- sum(e ^ 2) / max(n - p - 1, 1)
  cov_b <- sigma2 * XtX_inv
  se <- sqrt(pmax(diag(cov_b), 0))
  list(estimate = beta, se = se, rho = rho, sigma2 = sigma2, n = n,
       method = "SAR lag (ML, concentrated log-likelihood)")
}

# CANONICAL TEST  (with row-standardised path graph)

#' @rdname sarla
#' @keywords internal
#' @export
spatial_ar_lag <- sarla
