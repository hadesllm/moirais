# SPDX-License-Identifier: MIT OR Apache-2.0

#' Robinson partially-linear regression
#'
#' @param x Numeric parametric covariate vector or matrix.
#' @param y Numeric response vector.
#' @param z Numeric nonparametric covariate vector or matrix.
#' @param bandwidth Optional kernel bandwidth (Silverman default).
#' @return Named list with estimate, se, bandwidth, n, method.
#' @keywords internal
hrzp1 <- function(x, y, z, bandwidth = NULL) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  Z <- if (is.null(dim(z))) matrix(z, ncol = 1) else as.matrix(z)
  n <- length(y)
  if (n < 5 || nrow(X) != n || nrow(Z) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "PLR (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(Z[, 1]) else as.numeric(bandwidth)
  if (h <= 0) h <- max(.hrz_silverman(Z[, 1]), 1e-6)
  Zs <- if (ncol(Z) == 1) Z[, 1] else Z
  mY <- .hrz_nw_loo(Zs, y, h)
  mX <- sapply(seq_len(ncol(X)), function(j) .hrz_nw_loo(Zs, X[, j], h))
  if (is.null(dim(mX))) mX <- matrix(mX, ncol = ncol(X))
  rY <- y - mY; rX <- X - mX
  beta <- tryCatch(MASS::ginv(t(rX) %*% rX) %*% (t(rX) %*% rY),
                   error = function(e) rep(NA_real_, ncol(X)))
  resid <- rY - rX %*% beta
  bread <- MASS::ginv(t(rX) %*% rX)
  meat <- t(rX) %*% (rX * as.numeric(resid)^2)
  cov_m <- bread %*% meat %*% bread
  se <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = if (length(beta) == 1) as.numeric(beta) else as.numeric(beta),
       se = if (length(se) == 1) as.numeric(se) else as.numeric(se),
       bandwidth = h, n = n,
       method = "Robinson (1988) partially-linear regression")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzp1
#' @keywords internal
#' @export
horowitz_plr_estimator <- hrzp1
