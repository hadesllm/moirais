# SPDX-License-Identifier: MIT OR Apache-2.0

#' Max pooling forward pass (2D, single channel)
#'
#' R parity for \code{morie.fn.mxpol.maxpool_forward}.
#'
#' \deqn{y[i,j] = \max_{0 \le m,n < k} x[i s + m, j s + n]}
#'
#' @param x Numeric matrix \code{(H, W)}.
#' @param kernel_size Window size \code{k}.
#' @param stride Stride (default \code{kernel_size}, non-overlapping).
#' @return Named list \code{(y, estimate, argmax, output_shape, method)}.
#' @references Goodfellow et al. (2016), Deep Learning, Ch 9.3.
#' @export
mxpol_maxpool_forward <- function(x, kernel_size = 2L, stride = NULL) {
  x <- as.matrix(x)
  k <- as.integer(kernel_size)
  s <- if (is.null(stride)) k else as.integer(stride)
  H <- nrow(x); W <- ncol(x)
  if (H < k || W < k)
    stop(sprintf("Input (%d,%d) smaller than kernel %d", H, W, k))
  out_h <- (H - k) %/% s + 1L
  out_w <- (W - k) %/% s + 1L
  y <- matrix(0, out_h, out_w)
  argmax <- matrix(0L, out_h, out_w)
  for (i in seq_len(out_h)) {
    for (j in seq_len(out_w)) {
      i0 <- (i - 1L) * s + 1L
      j0 <- (j - 1L) * s + 1L
      block <- as.numeric(x[i0:(i0 + k - 1L), j0:(j0 + k - 1L)])
      y[i, j] <- max(block)
      argmax[i, j] <- which.max(block) - 1L
    }
  }
  list(y = y, estimate = y, argmax = argmax,
       output_shape = c(out_h, out_w),
       method = "MaxPool2D forward")
}

#' @rdname mxpol_maxpool_forward
#' @keywords internal
#' @export
maxpool_forward <- mxpol_maxpool_forward
