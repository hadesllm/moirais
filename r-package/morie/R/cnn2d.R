# SPDX-License-Identifier: MIT OR Apache-2.0

#' 2D convolution forward pass (cross-correlation, single channel)
#'
#' R parity for \code{morie.fn.cnn2d.conv2d_forward}.
#'
#' \deqn{y[i,j] = \sum_{m,n} w[m,n] x[i s + m, j s + n] + b}
#'
#' @param x Numeric matrix \code{(H, W)}.
#' @param w Numeric matrix \code{(K_h, K_w)}.
#' @param b Bias scalar.
#' @param stride Stride.
#' @param padding Zero-padding on each side.
#' @return Named list \code{(y, estimate, output_shape, method)}.
#' @references Goodfellow et al. (2016), Deep Learning, Ch 9.
#' @export
cnn2d_conv2d_forward <- function(x, w, b = 0, stride = 1L, padding = 0L) {
  x <- as.matrix(x); w <- as.matrix(w)
  if (padding > 0) {
    pad <- padding
    x <- rbind(matrix(0, pad, ncol(x)), x, matrix(0, pad, ncol(x)))
    x <- cbind(matrix(0, nrow(x), pad), x, matrix(0, nrow(x), pad))
  }
  H <- nrow(x); W <- ncol(x); Kh <- nrow(w); Kw <- ncol(w)
  if (H < Kh || W < Kw)
    stop(sprintf("Input (%d,%d) smaller than kernel (%d,%d)", H, W, Kh, Kw))
  out_h <- (H - Kh) %/% stride + 1L
  out_w <- (W - Kw) %/% stride + 1L
  y <- matrix(0, out_h, out_w)
  for (i in seq_len(out_h)) {
    for (j in seq_len(out_w)) {
      i0 <- (i - 1L) * stride + 1L
      j0 <- (j - 1L) * stride + 1L
      y[i, j] <- sum(w * x[i0:(i0 + Kh - 1L), j0:(j0 + Kw - 1L)]) + b
    }
  }
  list(y = y, estimate = y,
       output_shape = c(out_h, out_w),
       method = "Conv2D forward (cross-correlation)")
}

#' @rdname cnn2d_conv2d_forward
#' @keywords internal
#' @export
conv2d_forward <- cnn2d_conv2d_forward
