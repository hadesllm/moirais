# SPDX-License-Identifier: GPL-2.0-only
#' Penalised B-spline regression (P-splines; Eilers & Marx 1996)
#'
#' min_b ||y - Bb||^2 + lambda b' D' D b, with cubic B-spline basis and
#' second-order difference penalty.
#'
#' @param x numeric vector.
#' @param y numeric vector.
#' @param n_knots integer; default 20.
#' @param degree integer; default 3.
#' @param lam penalty (default 1).
#' @return list: coef, fitted, residuals, sse, r2, edf, lambda, n, method.
#' @importFrom splines bs
#' @keywords internal
pspln <- function(x, y, n_knots = 20L, degree = 3L, lam = 1) {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (n < degree + 2L || length(y) != n)
    return(list(estimate = NA_real_, n = n, method = "P-spline (n too small)"))
  knots <- seq(min(x), max(x), length.out = n_knots)
  B <- splines::bs(x, knots = knots[-c(1, length(knots))],
                   Boundary.knots = range(knots),
                   degree = degree, intercept = TRUE)
  B <- as.matrix(B); k <- ncol(B)
  D <- diff(diag(k), differences = 2L)
  BtB <- crossprod(B); BtY <- crossprod(B, y)
  coef <- as.numeric(solve(BtB + lam * crossprod(D), BtY))
  fitted <- as.numeric(B %*% coef)
  resid <- y - fitted
  sse <- sum(resid^2); sst <- sum((y - mean(y))^2)
  r2 <- if (sst > 0) 1 - sse / sst else NA_real_
  H <- B %*% solve(BtB + lam * crossprod(D), t(B))
  edf <- sum(diag(H))
  list(coef = coef, fitted = fitted, residuals = resid,
       sse = sse, r2 = as.numeric(r2),
       edf = as.numeric(edf), lambda = lam,
       estimate = mean(fitted),
       se = sqrt(sse / max(1, n - edf)) / sqrt(n),
       n = as.integer(n),
       method = "P-spline (Eilers & Marx 1996)")
}

# CANONICAL TEST
# set.seed(0); x <- seq(0, 1, length.out = 80); y <- sin(3*x) + rnorm(80, sd = 0.05)
# r <- pspln(x, y, lam = 0.1)
# stopifnot(r$r2 > 0.9)

#' @rdname pspln
#' @keywords internal
#' @export
penalized_spline <- pspln
