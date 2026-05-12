# SPDX-License-Identifier: GPL-2.0-only
#' Polynomial trend surface analysis (OLS).
#'
#' mu(s) = sum_k beta_k f_k(s); f_k are monomials up to degree `order`.
#'
#' @param x Numeric vector.
#' @param coords Coord matrix (n by d), d either 1 or 2.
#' @param order Polynomial order (0, 1, 2, or 3).
#' @return Named list: estimate, se, r2, order, n, method.
#' @references Schabenberger & Gotway (2005), Ch 2.
#' @export
sptrn <- function(x, coords, order = 2) {
  y <- as.numeric(x); n <- length(y)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  d <- ncol(coords)
  cols <- list(rep(1, n))
  if (d == 1) {
    s <- coords[, 1]
    for (k in seq_len(order)) cols[[length(cols) + 1]] <- s ^ k
  } else {
    s1 <- coords[, 1]; s2 <- coords[, 2]
    if (order >= 1) cols <- c(cols, list(s1, s2))
    if (order >= 2) cols <- c(cols, list(s1 ^ 2, s2 ^ 2, s1 * s2))
    if (order >= 3) cols <- c(cols, list(s1 ^ 3, s2 ^ 3, s1 ^ 2 * s2, s1 * s2 ^ 2))
    if (order >= 4) stop("trend_order > 3 not supported")
  }
  F_ <- do.call(cbind, cols)
  p <- ncol(F_)
  if (n < p) stop("need n >= ", p, " for order=", order)
  XtX <- crossprod(F_)
  beta <- as.numeric(solve(XtX, crossprod(F_, y)))
  e <- y - as.numeric(F_ %*% beta)
  sigma2 <- sum(e ^ 2) / max(n - p, 1)
  se <- sqrt(pmax(diag(sigma2 * solve(XtX)), 0))
  ss_tot <- sum((y - mean(y)) ^ 2)
  r2 <- if (ss_tot > 0) 1 - sum(e ^ 2) / ss_tot else 1
  list(estimate = beta, se = se, r2 = r2, order = as.integer(order),
       n = n, method = sprintf("Polynomial trend surface (order=%d, OLS)", order))
}

# CANONICAL TEST
# sptrn(c(1,2,3,4,5), matrix(0:4, ncol=1), order=1)$estimate  # c(1,1)

#' @rdname sptrn
#' @keywords internal
#' @export
spatial_trend_surface <- sptrn
