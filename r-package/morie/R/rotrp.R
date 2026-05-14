# SPDX-License-Identifier: MIT OR Apache-2.0

#' Rotary position embedding (RoPE)
#'
#' R parity for \code{morie.fn.rotrp.rotary_position_embedding}.
#'
#' Each pair (2i, 2i+1) is rotated by
#' \eqn{\theta_{pos, i} = pos / N^{2i/d}}.
#'
#' @param x Numeric matrix \code{(seq_len, d_model)}, \code{d_model} even.
#' @param base Frequency base (default 10000).
#' @return Named list \code{(y, estimate, angles, method)}.
#' @references Su et al. (2021), arXiv:2104.09864.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
rotrp_rotary_position_embedding <- function(x, base = 10000) {
  x <- as.matrix(x)
  seq_len <- nrow(x); d <- ncol(x)
  if (d %% 2L != 0L) stop(sprintf("d_model must be even for RoPE, got %d", d))
  half <- d %/% 2L
  pos <- seq_len(seq_len) - 1L
  idx <- seq_len(half) - 1L
  inv_freq <- 1 / base ^ ((2 * idx) / d)
  angles <- outer(pos, inv_freq)
  cos_a <- cos(angles); sin_a <- sin(angles)
  even <- seq(1L, d, by = 2L)
  odd <- seq(2L, d, by = 2L)
  x_even <- x[, even, drop = FALSE]
  x_odd  <- x[, odd, drop = FALSE]
  y_even <- x_even * cos_a - x_odd * sin_a
  y_odd  <- x_even * sin_a + x_odd * cos_a
  y <- x
  y[, even] <- y_even
  y[, odd]  <- y_odd
  list(y = y, estimate = y, angles = angles,
       method = "Rotary position embedding")
}

#' @rdname rotrp_rotary_position_embedding
#' @keywords internal
#' @export
rotary_position_embedding <- rotrp_rotary_position_embedding
