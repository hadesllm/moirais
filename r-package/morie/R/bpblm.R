# SPDX-License-Identifier: MIT OR Apache-2.0

#' Bits per byte (Gao 2020)
#'
#' @param x Numeric vector of per-token NLLs in nats.
#' @param n_bytes Optional integer byte count; defaults to length(x).
#' @return Named list with value, nll_nats, n_tokens, n_bytes, method.
#' @keywords internal
bits_per_byte <- function(x, n_bytes = NULL) {
  nll <- as.numeric(x)
  if (!length(nll)) stop("Need at least one token NLL")
  total <- sum(nll); nb <- if (is.null(n_bytes)) length(nll) else as.integer(n_bytes)
  if (nb <= 0) stop("n_bytes must be > 0")
  list(value = total / (nb * log(2)),
       nll_nats = total, n_tokens = length(nll), n_bytes = nb,
       method = "BPB")
}
