# SPDX-License-Identifier: MIT OR Apache-2.0

#' Sinusoidal positional encoding
#'
#' R parity for \code{morie.fn.posab.positional_encoding_abs}.
#'
#' \deqn{\mathrm{PE}(pos, 2i)   = \sin(pos / N^{2i/d})}
#' \deqn{\mathrm{PE}(pos, 2i+1) = \cos(pos / N^{2i/d})}
#'
#' @param seq_len Sequence length.
#' @param d_model Embedding dimension.
#' @param base Frequency base (default 10000).
#' @return Named list \code{(PE, estimate, seq_len, d_model, method)}.
#' @references Vaswani et al. (2017), NeurIPS.
#' @export
posab_positional_encoding_abs <- function(seq_len, d_model, base = 10000) {
  seq_len <- as.integer(seq_len); d_model <- as.integer(d_model)
  if (seq_len <= 0 || d_model <= 0)
    stop("seq_len and d_model must be > 0.")
  pos <- matrix(seq_len(seq_len) - 1L, ncol = 1L)
  i <- matrix(0:(d_model - 1L), nrow = 1L)
  div_term <- base ^ ((2 * (i %/% 2L)) / d_model)
  angles <- pos[, rep(1L, d_model)] / div_term[rep(1L, seq_len), ]
  PE <- angles
  even <- seq(1L, d_model, by = 2L)
  odd <- seq(2L, d_model, by = 2L)
  PE[, even] <- sin(angles[, even, drop = FALSE])
  PE[, odd]  <- cos(angles[, odd, drop = FALSE])
  list(PE = PE, estimate = PE, seq_len = seq_len, d_model = d_model,
       method = "Sinusoidal positional encoding")
}

#' @rdname posab_positional_encoding_abs
#' @keywords internal
#' @export
positional_encoding_abs <- posab_positional_encoding_abs
