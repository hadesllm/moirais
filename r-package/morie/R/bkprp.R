# SPDX-License-Identifier: GPL-2.0-only

#' Backpropagation gradient computation (1-layer MSE)
#'
#' R parity for \code{morie.fn.bkprp.backpropagation}.
#'
#' For \eqn{L = \tfrac{1}{2n}\sum (a - y)^2} with
#' \eqn{a = \sigma(W x + b)}:
#' \deqn{\partial L / \partial W = \delta x^\top, \quad
#'       \delta = (a - y) \odot \sigma'(z)}
#'
#' @param x Numeric matrix \code{(batch, n_in)} or vector.
#' @param y Numeric matrix \code{(batch, n_out)} or vector.
#' @param w Weight matrix; defaults to \code{diag(n_out, n_in)}.
#' @param b Bias vector; defaults to zeros.
#' @param activation One of \code{"sigmoid"}, \code{"tanh"},
#'   \code{"relu"}, \code{"identity"}.
#' @return Named list \code{(loss, estimate, dW, db, dx, a, z, method)}.
#' @references Rumelhart, Hinton & Williams (1986); Goodfellow et al. (2016).
#' @export
bkprp_backpropagation <- function(x, y, w = NULL, b = NULL,
                                  activation = "sigmoid") {
  x <- as.matrix(x); y <- as.matrix(y)
  n_in <- ncol(x); n_out <- ncol(y)
  if (is.null(w)) w <- diag(1, n_out, n_in)
  if (is.null(b)) b <- rep(0, n_out)
  w <- as.matrix(w); b <- as.numeric(b)
  z <- sweep(x %*% t(w), 2L, b, "+")
  a <- .bkprp_sigma(z, activation)
  dsig <- .bkprp_sigma_prime(z, activation, a)
  batch <- nrow(x)
  diff <- a - y
  loss <- 0.5 * sum(diff * diff) / batch
  delta <- diff * dsig
  dW <- t(delta) %*% x / batch
  db <- colSums(delta) / batch
  dx <- delta %*% w / batch
  list(loss = loss, estimate = loss, dW = dW, db = db, dx = dx, a = a, z = z,
       method = "Backpropagation gradient computation")
}

.bkprp_sigma <- function(z, activation) {
  switch(activation,
    "identity" = z, "linear" = z, "none" = z,
    "sigmoid" = 1 / (1 + exp(-z)),
    "tanh" = tanh(z),
    "relu" = pmax(0, z),
    stop(sprintf("Unknown activation: %s", activation))
  )
}

.bkprp_sigma_prime <- function(z, activation, a) {
  switch(activation,
    "identity" = matrix(1, nrow(z), ncol(z)),
    "linear" = matrix(1, nrow(z), ncol(z)),
    "none" = matrix(1, nrow(z), ncol(z)),
    "sigmoid" = a * (1 - a),
    "tanh" = 1 - a * a,
    "relu" = (z > 0) * 1.0,
    stop(sprintf("Unknown activation: %s", activation))
  )
}

#' @rdname bkprp_backpropagation
#' @keywords internal
#' @export
backpropagation <- bkprp_backpropagation
