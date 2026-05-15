# SPDX-License-Identifier: AGPL-3.0-or-later
#' Thin-plate spline regression (Duchon 1977)
#'
#' Fits f(x) = sum a_i phi(||x - x_i||) + beta_0 + beta' x with radial
#' basis phi(r) = r^2 log(r).  Solves the augmented saddle-point system.
#' For larger problems, prefer \code{mgcv::gam(y ~ s(x, bs="tp"))}.
#'
#' @param x numeric vector or matrix of predictors.
#' @param y numeric outcome.
#' @param lam smoothing penalty; 0 = interpolation.
#' @return list: a, beta, fitted, residuals, sse, r2, lambda, n, d, method.
#' @keywords internal
tpspn <- function(x, y, lam = 0) {
  if (!is.matrix(x)) x <- matrix(x, ncol = 1)
  y <- as.numeric(y); n <- nrow(x); d <- ncol(x)
  if (n < d + 2L || length(y) != n)
    return(list(estimate = NA_real_, n = n, method = "TPS (n too small)"))
  R <- as.matrix(stats::dist(x))
  phi <- function(r) ifelse(r > 0, r^2 * log(r), 0)
  K <- phi(R) + lam * diag(n)
  Tmat <- cbind(1, x)
  A <- rbind(cbind(K, Tmat),
             cbind(t(Tmat), matrix(0, d + 1, d + 1)))
  rhs <- c(y, rep(0, d + 1))
  sol <- as.numeric(MASS::ginv(A) %*% rhs)
  a <- sol[seq_len(n)]
  beta <- sol[(n + 1):length(sol)]
  fitted <- as.numeric(K %*% a + Tmat %*% beta)
  resid <- y - fitted
  sse <- sum(resid^2); sst <- sum((y - mean(y))^2)
  r2 <- if (sst > 0) 1 - sse / sst else NA_real_
  list(a = a, beta = beta, fitted = fitted, residuals = resid,
       sse = sse, r2 = as.numeric(r2), lambda = lam,
       estimate = mean(fitted),
       n = as.integer(n), d = as.integer(d),
       method = "Thin-plate spline (Duchon 1977)")
}

# CANONICAL TEST
# set.seed(0); xx <- matrix(runif(80), ncol = 2); yy <- xx[,1] + xx[,2]
# r <- tpspn(xx, yy, lam = 1e-8)
# stopifnot(r$r2 > 0.99)

#' @rdname tpspn
#' @keywords internal
#' @export
thin_plate_spline <- tpspn
