# SPDX-License-Identifier: MIT OR Apache-2.0

#' GP posterior mean with Matern kernel.
#'
#' @param x Numeric vector or matrix of input points.
#' @param y Numeric response vector.
#' @param nu Matern smoothness parameter (default 1.5).
#' @param length_scale Optional kernel length-scale.
#' @param sigma_f Numeric signal sd (default 1).
#' @param noise Optional observation noise sd.
#' @param x_star Optional matrix of prediction points (defaults to x).
#' @return Named list with estimate, se, mu, sd, length_scale, nu, noise, n, method.
#' @export
ghosal_gp_matern <- function(x, y, nu = 1.5, length_scale = NULL,
                              sigma_f = 1.0, noise = NULL, x_star = NULL) {
  if (is.vector(x)) x <- matrix(as.numeric(x), ncol = 1L) else x <- as.matrix(x)
  y <- as.numeric(y); n <- nrow(x)
  if (is.null(x_star)) {
    x_star <- x
  } else if (is.vector(x_star)) {
    x_star <- matrix(as.numeric(x_star), ncol = 1L)
  } else {
    x_star <- as.matrix(x_star)
  }
  sq <- pmax(.gh_pairwise_sq(x), 0)
  if (is.null(length_scale)) {
    d <- sqrt(sq[upper.tri(sq)])
    length_scale <- if (length(d)) max(stats::median(d[d > 0]), 1e-3) else 1
  }
  if (is.null(noise)) noise <- max(0.1 * stats::sd(y), 1e-3)
  kernel <- function(a, b) {
    sq_ab <- pmax(.gh_pairwise_sq(a, b), 0)
    r <- sqrt(sq_ab)
    if (isTRUE(all.equal(nu, 0.5)))  return(sigma_f^2 * exp(-r / length_scale))
    if (isTRUE(all.equal(nu, 1.5))) {
      t <- sqrt(3) * r / length_scale
      return(sigma_f^2 * (1 + t) * exp(-t))
    }
    if (isTRUE(all.equal(nu, 2.5))) {
      t <- sqrt(5) * r / length_scale
      return(sigma_f^2 * (1 + t + t^2/3) * exp(-t))
    }
    rr <- pmax(r, 1e-12); z <- sqrt(2*nu) * rr / length_scale
    coef <- sigma_f^2 * 2^(1 - nu) / gamma(nu)
    K <- coef * z^nu * besselK(z, nu)
    K[r < 1e-12] <- sigma_f^2
    K
  }
  K <- kernel(x, x) + noise^2 * diag(n)
  K_s <- kernel(x_star, x)
  K_ss_diag <- rep(sigma_f^2, nrow(x_star))
  L <- chol(K + 1e-8 * diag(n))
  alpha_ <- backsolve(L, forwardsolve(t(L), y))
  mu <- as.numeric(K_s %*% alpha_)
  v <- forwardsolve(t(L), t(K_s))
  var <- K_ss_diag - colSums(v^2)
  sd_ <- sqrt(pmax(var, 0))
  list(estimate = mean(mu), se = mean(sd_), mu = mu, sd = sd_,
       length_scale = length_scale, nu = nu, noise = noise, n = n,
       method = "GP regression (Matern kernel)")
}
