# SPDX-License-Identifier: GPL-2.0-only
#' Universal kriging with polynomial trend.
#'
#' Z(s) = mu(s) + delta(s), mu(s) = sum_k beta_k f_k(s).
#'
#' @param x Numeric vector.
#' @param coords Numeric coordinate matrix (n by d).
#' @param target Numeric matrix (m by d) or vector (d,).
#' @param model Covariance: "exponential", "gaussian", "spherical".
#' @param nugget,sill,range_ Variogram parameters.
#' @param trend_order 0, 1, or 2.
#' @return Named list: estimate, se, n, method.
#' @references Schabenberger & Gotway (2005), Ch 4.
#' @export
ukrig <- function(x, coords, target, model = "exponential",
                  nugget = 0, sill = 1, range_ = 1, trend_order = 1) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  target <- if (is.matrix(target)) target else
    matrix(as.numeric(unlist(target)), ncol = ncol(coords))
  if (nrow(coords) != n) stop("coords rows must match length(x)")
  if (ncol(target) != ncol(coords)) stop("target dim mismatch")
  c0 <- nugget; c1 <- sill - nugget; a <- range_
  cov_fn <- function(h) {
    switch(model,
           exponential = c1 * exp(-h / a) + ifelse(h == 0, c0, 0),
           gaussian    = c1 * exp(-(h ^ 2) / (a ^ 2)) + ifelse(h == 0, c0, 0),
           spherical   = ifelse(h <= a,
                                c1 * (1 - 1.5 * h / a + 0.5 * (h / a) ^ 3),
                                0) + ifelse(h == 0, c0, 0),
           stop("unknown model"))
  }
  trend_design <- function(C) {
    C <- if (is.matrix(C)) C else matrix(C, ncol = ncol(coords))
    n_ <- nrow(C)
    ones <- matrix(1, n_, 1)
    if (trend_order == 0) return(ones)
    if (trend_order == 1) return(cbind(ones, C))
    if (trend_order == 2) {
      sq <- C ^ 2
      cross <- if (ncol(C) >= 2) C[, 1] * C[, 2] else NULL
      return(cbind(ones, C, sq, cross))
    }
    stop("trend_order must be 0, 1, or 2")
  }
  Dnn <- as.matrix(stats::dist(coords))
  Cmat <- cov_fn(Dnn)
  F_ <- trend_design(coords)
  p <- ncol(F_)
  K <- matrix(0, n + p, n + p)
  K[1:n, 1:n] <- Cmat
  K[1:n, (n + 1):(n + p)] <- F_
  K[(n + 1):(n + p), 1:n] <- t(F_)
  total_var <- c0 + c1
  m <- nrow(target)
  ests <- numeric(m); ses <- numeric(m)
  for (k in seq_len(m)) {
    d0 <- sqrt(colSums((t(coords) - target[k, ]) ^ 2))
    c_vec <- cov_fn(d0)
    f0 <- as.numeric(trend_design(matrix(target[k, ], 1)))
    rhs <- c(c_vec, f0)
    sol <- tryCatch(solve(K, rhs),
                    error = function(e) qr.solve(K, rhs))
    lam <- sol[1:n]
    ests[k] <- sum(lam * x)
    ses[k] <- sqrt(max(total_var - sum(sol * rhs), 0))
  }
  list(estimate = if (m == 1) ests[1] else ests,
       se = if (m == 1) ses[1] else ses,
       n = n,
       method = sprintf("Universal kriging (%s, trend_order=%d)", model, trend_order))
}

# CANONICAL TEST
# ukrig(c(1,2,3,4,5), matrix(0:4, ncol=1), matrix(2.5, 1, 1),
#       trend_order=1)$estimate   # ~ 3.5

#' @rdname ukrig
#' @keywords internal
#' @export
universal_kriging <- ukrig
