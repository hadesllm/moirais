# SPDX-License-Identifier: GPL-2.0-only

#' 1D convolution forward pass (cross-correlation, valid padding)
#'
#' R parity for \code{morie.fn.cnn1d.conv1d_forward}.
#'
#' \deqn{y[i] = \sum_{k=0}^{K-1} w[k] x[i\,s + k] + b}
#'
#' @param x Numeric vector.
#' @param w Numeric kernel vector.
#' @param b Bias scalar.
#' @param stride Stride (default 1).
#' @param padding Zero-padding on each side (default 0).
#' @return Named list \code{(y, estimate, output_length, method)}.
#' @references Goodfellow et al. (2016), Deep Learning, Ch 9.
#' @export
cnn1d_conv1d_forward <- function(x, w, b = 0, stride = 1L, padding = 0L) {
  x <- as.numeric(x); w <- as.numeric(w)
  if (padding > 0) x <- c(rep(0, padding), x, rep(0, padding))
  K <- length(w); N <- length(x)
  if (N < K) stop(sprintf("Input length %d < kernel length %d", N, K))
  out_len <- (N - K) %/% stride + 1L
  y <- numeric(out_len)
  for (j in seq_len(out_len)) {
    i0 <- (j - 1L) * stride + 1L
    y[j] <- sum(w * x[i0:(i0 + K - 1L)]) + b
  }
  list(y = y, estimate = y, output_length = out_len,
       method = "Conv1D forward (cross-correlation)")
}

#' @rdname cnn1d_conv1d_forward
#' @keywords internal
#' @export
conv1d_forward <- cnn1d_conv1d_forward
