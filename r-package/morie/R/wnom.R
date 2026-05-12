# SPDX-License-Identifier: GPL-2.0-only

#' W-NOMINATE Gaussian-utility log-likelihood (Armstrong Ch 3)
#'
#' Computes the Poole-Rosenthal NOMINATE log-likelihood and the
#' geometric mean probability (GMP) given legislator ideal points and
#' yea/nay outcome locations:
#'   U_ijy = beta * exp(-0.5 sum_k w_k^2 (x_ik - z_yjk)^2) + epsilon
#'   P(yea) = Phi(U_yea - U_nay).
#'
#' @param votes (n_leg by n_votes) matrix; 1 = yea, 0 = nay, NA = miss.
#' @param x Legislator ideal points (n_leg by p).
#' @param z_yea Yea outcome locations (n_votes by p).
#' @param z_nay Nay outcome locations (n_votes by p).
#' @param beta Signal-to-noise (default 15).
#' @param w Optional dimension salience weights (length p; default 1).
#' @return Named list with `loglik`, `GMP`, `n_correct`, `n_total`,
#'   `method`.
#' @export
wnom <- function(votes, x, z_yea, z_nay, beta = 15, w = NULL) {
  X <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1L)
  Zy <- if (is.matrix(z_yea)) z_yea else matrix(as.numeric(z_yea), ncol = 1L)
  Zn <- if (is.matrix(z_nay)) z_nay else matrix(as.numeric(z_nay), ncol = 1L)
  p <- ncol(X)
  if (is.null(w)) w <- rep(1, p)
  V <- if (is.matrix(votes)) votes else matrix(as.numeric(votes), nrow = nrow(X))
  n_leg <- nrow(X); n_votes <- nrow(Zy)
  dy <- array(0, dim = c(n_leg, n_votes))
  dn <- array(0, dim = c(n_leg, n_votes))
  for (k in seq_len(p)) {
    dxy <- outer(X[, k], Zy[, k], FUN = "-")
    dxn <- outer(X[, k], Zn[, k], FUN = "-")
    dy <- dy + (w[k] * dxy)^2
    dn <- dn + (w[k] * dxn)^2
  }
  U_y <- beta * exp(-0.5 * dy)
  U_n <- beta * exp(-0.5 * dn)
  P <- stats::pnorm(U_y - U_n)
  P <- pmin(pmax(P, 1e-10), 1 - 1e-10)
  mask <- !is.na(V)
  ll <- sum(ifelse(mask & V == 1, log(P), 0)) +
        sum(ifelse(mask & V == 0, log(1 - P), 0))
  pred <- (P > 0.5) * 1L
  n_correct <- sum(mask & (pred == V))
  n_total <- sum(mask)
  GMP <- if (n_total > 0L) n_correct / n_total else 0
  list(loglik = ll, GMP = GMP, n_correct = n_correct,
       n_total = n_total, method = "wnominate_estimate")
}

#' @keywords internal
#' @rdname wnom
#' @export
wnominate_estimate <- wnom

#' @rdname wnom
#' @keywords internal
#' @export
wnominate <- wnom
