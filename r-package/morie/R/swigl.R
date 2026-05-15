# SPDX-License-Identifier: AGPL-3.0-or-later

#' SwiGLU activation (Shazeer 2020)
#'
#' @param x Input matrix.
#' @param W Optional gate-projection weights.
#' @param V Optional up-projection weights.
#' @param b Optional gate bias.
#' @param c Optional up-projection bias.
#' @return Named list with tensor, gate, up, method.
#' @keywords internal
swiglu_activation <- function(x, W = NULL, V = NULL, b = NULL, c = NULL) {
  if (is.null(W) && is.null(V)) {
    d_out <- ncol(as.matrix(x))
    W <- diag(d_out); V <- diag(d_out)
  } else if (xor(is.null(W), is.null(V))) {
    stop("Provide both W and V or neither")
  }
  if (is.null(b)) b <- rep(0, ncol(W))
  if (is.null(c)) c <- rep(0, ncol(V))
  xm <- as.matrix(x)
  gate <- sweep(xm %*% W, 2L, b, "+")
  silu_gate <- gate * (1 / (1 + exp(-gate)))
  up <- sweep(xm %*% V, 2L, c, "+")
  out <- silu_gate * up
  list(tensor = out, gate = silu_gate, up = up, method = "SwiGLU")
}
