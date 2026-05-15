# SPDX-License-Identifier: AGPL-3.0-or-later

#' Causal autoregressive mask (Radford 2019)
#'
#' @param x Sequence length (integer) or tensor whose second-to-last
#'   dim is the sequence length.
#' @return Named list with tensor, n, method.
#' @keywords internal
causal_attention_mask <- function(x) {
  n <- if (length(x) == 1L && is.numeric(x)) as.integer(x)
       else if (!is.null(dim(x))) dim(x)[length(dim(x)) - 1L]
       else length(x)
  M <- matrix(0, n, n)
  M[upper.tri(M)] <- -Inf
  list(tensor = M, n = n, method = "causal-mask")
}
