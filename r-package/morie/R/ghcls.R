# SPDX-License-Identifier: MIT OR Apache-2.0

#' Probit-GP classifier (Laplace approximation).
#'
#' @param x Numeric matrix of features.
#' @param y Numeric binary labels (0/1).
#' @param length_scale Optional kernel length-scale.
#' @param sigma_f Numeric signal sd (default 1).
#' @param n_iter Integer maximum Laplace iterations (default 300).
#' @param seed Integer RNG seed (default 0).
#' @return Named list with estimate, p_hat, accuracy, length_scale, n, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ghosal_np_classification <- function(x, y, length_scale = NULL,
                                       sigma_f = 1.0, n_iter = 300, seed = 0) {
  set.seed(seed)
  x <- as.matrix(x); y <- as.numeric(y); n <- nrow(x)
  y_pm <- 2 * y - 1
  sq <- pmax(.gh_pairwise_sq(x), 0)
  if (is.null(length_scale)) {
    d <- sqrt(sq[upper.tri(sq)])
    length_scale <- if (length(d)) max(stats::median(d[d > 0]), 1e-3) else 1
  }
  K <- sigma_f^2 * exp(-sq / (2*length_scale^2)) + 1e-6 * diag(n)
  f <- rep(0, n)
  for (it in seq_len(n_iter)) {
    z <- y_pm * f
    phi <- stats::dnorm(z); Phi <- pmin(pmax(stats::pnorm(z), 1e-12), 1-1e-12)
    grad_ll <- y_pm * phi / Phi
    W <- pmax((phi/Phi) * (phi/Phi + z), 1e-8)
    sW <- sqrt(W)
    # B = I + diag(sW) %*% K %*% diag(sW)
    B <- diag(n) + (sW %o% sW) * K
    Lf <- tryCatch(chol(B), error = function(e) NULL)
    if (is.null(Lf)) break
    b <- W * f + grad_ll
    a <- b - sW * backsolve(Lf, forwardsolve(t(Lf), sW * (K %*% b)))
    f_new <- as.numeric(K %*% a)
    if (max(abs(f_new - f)) < 1e-6) { f <- f_new; break }
    f <- f_new
  }
  p_hat <- stats::pnorm(f); pred <- as.integer(p_hat >= 0.5)
  accuracy <- mean(pred == y)
  list(estimate = mean(p_hat), p_hat = p_hat, accuracy = accuracy,
       length_scale = length_scale, n = n,
       method = "Probit-link GP classifier (Laplace)")
}
