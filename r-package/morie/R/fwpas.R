# SPDX-License-Identifier: MIT OR Apache-2.0

#' Dense layer forward pass
#'
#' Pure-base-R parity for \code{morie.fn.fwpas.forward_pass_dense}.
#'
#' \deqn{z = W x + b, \quad a = \sigma(z)}
#'
#' @param x Numeric vector (single input) or matrix (rows = samples).
#' @param w Weight matrix shape \code{(n_out, n_in)}.
#' @param b Bias vector length \code{n_out}.
#' @param activation One of \code{"sigmoid"}, \code{"tanh"},
#'   \code{"relu"}, \code{"softmax"}, \code{"identity"}.
#' @return Named list with components \code{z}, \code{a}, \code{estimate}
#'   (= \code{a}), \code{activation}, \code{method}.
#' @references Goodfellow, Bengio & Courville (2016), Deep Learning, Ch 6.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
fwpas_forward_pass_dense <- function(x, w, b, activation = "sigmoid") {
  x <- as.matrix(x)
  w <- as.matrix(w)
  b <- as.numeric(b)
  if (ncol(x) == 1L && nrow(x) == length(b) + (ncol(w) - 1L)) {
    # 1-D input passed as column vector
    x <- t(x)
  } else if (ncol(x) != ncol(w)) {
    # try transpose if user passed (n_in,)
    if (nrow(x) == ncol(w)) x <- t(x)
  }
  z <- x %*% t(w)
  z <- sweep(z, 2L, b, "+")
  a <- switch(activation,
    "identity" = z,
    "linear"   = z,
    "none"     = z,
    "sigmoid"  = 1 / (1 + exp(-z)),
    "tanh"     = tanh(z),
    "relu"     = pmax(0, z),
    "softmax"  = {
      ez <- exp(z - apply(z, 1L, max))
      sweep(ez, 1L, rowSums(ez), "/")
    },
    stop(sprintf("Unknown activation: %s", activation))
  )
  list(z = z, a = a, estimate = a, activation = activation,
       method = "Dense layer forward pass")
}

#' @rdname fwpas_forward_pass_dense
#' @keywords internal
#' @export
forward_pass_dense <- fwpas_forward_pass_dense
