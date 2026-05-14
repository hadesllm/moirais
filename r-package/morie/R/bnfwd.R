# SPDX-License-Identifier: MIT OR Apache-2.0

#' Batch normalization forward pass
#'
#' R parity for \code{morie.fn.bnfwd.batch_norm_forward}.
#'
#' \deqn{y_i = \gamma\,\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta}
#'
#' @param x Numeric matrix shape \code{(batch, features)}.
#' @param gamma Scale vector (default ones).
#' @param beta Shift vector (default zeros).
#' @param eps Numerical stability (default \code{1e-5}).
#' @return Named list \code{(y, estimate, x_hat, mu, var, eps, method)}.
#' @references Ioffe & Szegedy (2015), ICML.
#' @export
bnfwd_batch_norm_forward <- function(x, gamma = NULL, beta = NULL,
                                     eps = 1e-5) {
  x <- as.matrix(x)
  # axis=0 (batch) -> per-feature stats
  mu <- colMeans(x)
  var <- apply(x, 2L, function(v) mean((v - mean(v))^2))
  if (is.null(gamma)) gamma <- rep(1, ncol(x))
  if (is.null(beta)) beta <- rep(0, ncol(x))
  x_hat <- sweep(x, 2L, mu, "-")
  x_hat <- sweep(x_hat, 2L, sqrt(var + eps), "/")
  y <- sweep(x_hat, 2L, gamma, "*")
  y <- sweep(y, 2L, beta, "+")
  list(y = y, estimate = y, x_hat = x_hat, mu = mu, var = var, eps = eps,
       method = "Batch normalization forward")
}

#' @rdname bnfwd_batch_norm_forward
#' @keywords internal
#' @export
batch_norm_forward <- bnfwd_batch_norm_forward
